from __future__ import annotations

import os
import copy
import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union

os.environ['RWKV_JIT_ON'] = '1'
os.environ["RWKV_CUDA_ON"] = '1'  # '1' to compile CUDA kernel

from rwkv.model import RWKV
from rwkv.utils import PIPELINE


@dataclass
class EvalArgs:
    question_ids: list[int]
    answer_ids: list[int]
    control_coeff: float
    answer_length: int


class ControllableRWKV(ABC):

    def __init__(self, model_path: str, strategy: str = 'cuda fp16'):
        self.model = RWKV(model=model_path, strategy=strategy)
        self.pipeline = PIPELINE(self.model, r"rwkv_vocab_v20230424")
        self.tokenizer = self.pipeline.tokenizer

    @staticmethod
    def _modify_state(state: list, direction: list, coeff: float, layer_rule: callable = lambda x: len(x.shape) != 1):
        if state is None:
            print('state is empty')
            return state
        for i in range(len(state)):
            if layer_rule(state[i]):
                state[i] += direction[i] * coeff
        return state

    @staticmethod
    def _top_kp_sampling(logits, temperature=1.0, k=10, p=0.5):
        """
        Apply top-k and top-p (nucleus) sampling on logits at given temperature using PyTorch.

        :param logits: PyTorch tensor containing logits from a language model's output layer.
        :param temperature: Temperature for softmax. Lower for less randomness.
        :param k: The number of highest probability choices to sample from (top-k).
        :param p: The cumulative probability threshold for sampling (top-p).
        :return: Index of the selected token.
        """
        # Apply temperature scaling
        logits = logits / temperature

        # Compute probabilities using softmax
        probs = torch.softmax(logits, dim=-1)

        # Top-k filtering
        if k > 0:
            top_k_values, top_k_indices = torch.topk(probs, k)
            mask = torch.ones_like(probs, dtype=torch.bool)
            mask[top_k_indices] = False
            probs[mask] = 0

            # Re-normalize probabilities after top-k
            probs /= probs.sum()

        # Re-normalize probabilities after top-k
        probs /= probs.sum()

        # Top-p filtering
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        # Remove tokens with cumulative probability above the threshold p
        sorted_indices_to_remove = cumulative_probs > p
        # Shift the indices to the right to keep the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        # Set probabilities of removed indices to 0
        probs[sorted_indices[sorted_indices_to_remove]] = 0

        # Re-normalize probabilities after top-p
        probs /= probs.sum()

        # Sample from the remaining distribution
        token_index = torch.multinomial(probs, 1).item()  # Sample one index

        return token_index

    @staticmethod
    def _calculate_average_direction(direction_list):
        new_direction = []
        for hidden_idx in range(len(direction_list[0])):
            accumulated = []
            for direction in direction_list:
                accumulated.append(direction[hidden_idx])
            mean = torch.mean(torch.stack(accumulated), dim=0)
            new_direction.append(copy.deepcopy(mean))
        return new_direction

    def forward(self, input_ids: int, state: Union[list, None]):
        return self.model.forward([input_ids], state)

    def update(self, pos_ids_list: list, neg_ids_list: list):

        if len(pos_ids_list) == 0 or len(neg_ids_list) == 0:
            raise ValueError("The 'ids_list' should not be empty.")

        if len(pos_ids_list) != len(neg_ids_list):
            raise ValueError("The length of pos_prompt_list and neg_prompt_list must be the same.")

        self._implemented_update(pos_ids_list, neg_ids_list)

    @abstractmethod
    def _implemented_update(self, pos_ids_list: list[int], neg_ids_list: list[int]):
        pass

    def run_loglikelihood(self, eval_args: EvalArgs) -> tuple[float, float]:

        if not eval_args.question_ids:
            raise ValueError("question_ids should not be empty.")

        if not eval_args.answer_ids:
            raise ValueError("answer_ids should not be empty.")

        return self._implemented_run_loglikelihood(eval_args)

    @abstractmethod
    def _implemented_run_loglikelihood(self, eval_args: EvalArgs) -> tuple[float, float]:
        pass

    @abstractmethod
    def generate(self, input_ids: list[int],
                 max_length: int,
                 top_k: int,
                 top_p: float,
                 temperature: float,
                 control_coeff: float) -> list[int]:
        """Generate a sequence of tokens using the model."""
        pass


class VectorControlRWKV(ControllableRWKV):
    def __init__(self, model_path: str, strategy: str = 'cuda fp16'):
        super().__init__(model_path, strategy)

        self.direction = None

    @torch.no_grad()
    def _implemented_update(self, pos_ids_list: list[int], neg_ids_list: list[int]):

        direction_list = []
        for pos_ids, neg_ids in zip(pos_ids_list, neg_ids_list):

            pos_state = self.model.forward(pos_ids, None)[1]
            neg_state = self.model.forward(neg_ids, None)[1]

            direction = [p - n for p, n in zip(pos_state, neg_state)]

            direction_list.append(direction)

        self.direction = self._calculate_average_direction(direction_list)

    @torch.no_grad()
    def generate(self,
                 input_ids: list[int],
                 max_length: int = 20,
                 top_k: int = 10,
                 top_p: float = 0.5,
                 temperature: float = 1.0,
                 control_coeff: float = None) -> list[int]:
        """
        :param top_k:
        :param control_coeff:
        :param input_ids:
        :param max_length:
        :param top_p:
        :param temperature:
        :return:
        """

        state = None
        generated = []

        # prefill
        for input_id in input_ids[:-1]:
            _, state = self.forward(input_id, state)

        # modify state
        state = self._modify_state(state, self.direction, control_coeff)

        # generate
        input_id = input_ids[-1]

        for _ in range(max_length):
            logits, state = self.forward(input_id, state)

            input_id = self._top_kp_sampling(logits, temperature=temperature, k=top_k, p=top_p)

            if input_id == 0:
                break

            generated.append(input_id)

        return generated

    @torch.no_grad()
    def _implemented_run_loglikelihood(self, eval_args: EvalArgs) -> tuple[float, float]:

        state = self.model.forward(eval_args.question_ids, None)[1]

        # We only add control vectors at the end of the question
        state = self._modify_state(state, self.direction, eval_args.control_coeff)

        logits = []

        for token_pos in range(len(eval_args.answer_ids[:-1])):

            logit, state = self.forward(eval_args.answer_ids[token_pos], state)
            logits.append(logit)

        # Move the entire list of logits to the CPU if necessary, after the loop
        logits = torch.stack(logits).cpu()
        distributions = torch.softmax(logits, dim=-1)
        target_ids = eval_args.answer_ids[1:]

        # Use tensor operations to gather target probabilities
        target_probs = distributions[range(len(target_ids)), target_ids]

        loglikelihood = torch.log(target_probs)

        # Calculate the sum of the log likelihoods where the target mask is True, then normalize
        target_loglikelihood = loglikelihood.sum().item()

        # target_loglikelihood_norm = target_loglikelihood / (len(eval_args.answer_ids) - 1)
        target_loglikelihood_norm = target_loglikelihood / eval_args.answer_length

        return target_loglikelihood, target_loglikelihood_norm


class ContrastControlRWKV(ControllableRWKV):
    def __init__(self, model_path: str, strategy: str = 'cuda fp16'):
        super().__init__(model_path, strategy)

        self.pos_pre_state_list = None
        self.neg_pre_state_list = None

    @torch.no_grad()
    def _implemented_update(self, pos_ids_list: list[str], neg_ids_list: list[str]):

        self.pos_pre_state_list = []
        self.neg_pre_state_list = []

        for pos_prompt_ids, neg_prompt_ids in zip(pos_ids_list, neg_ids_list):
            self.pos_pre_state_list.append(self.model.forward(pos_prompt_ids, None)[1])
            self.neg_pre_state_list.append(self.model.forward(neg_prompt_ids, None)[1])

    def get_direction_from_state_list(self, pos_state_list: list, neg_state_list: list):
        direction_list = []
        for pos_state, neg_state in zip(pos_state_list, neg_state_list):
            direction = [p - n for p, n in zip(pos_state, neg_state)]
            direction_list.append(direction)

        avg_direction = self._calculate_average_direction(direction_list)

        return avg_direction

    @torch.no_grad()
    def generate(self,
                 input_ids: list[int],
                 max_length: int = 20,
                 top_k: int = 10,
                 top_p: float = 0.5,
                 temperature: float = 1.0,
                 control_coeff: float = 0) -> list[int]:
        generated = []

        # prefill
        pos_state_list = copy.deepcopy(self.pos_pre_state_list)
        neg_state_list = copy.deepcopy(self.neg_pre_state_list)

        normal_state = None
        if len(input_ids) > 1:
            input_tokens = input_ids[:-1]
            normal_state = self.model.forward(input_tokens, normal_state)[1]
            for idx in range(len(pos_state_list)):
                pos_state_list[idx] = self.model.forward(input_tokens, pos_state_list[idx])[1]
                neg_state_list[idx] = self.model.forward(input_tokens, neg_state_list[idx])[1]

        # generate
        input_id = input_ids[-1]

        for _ in range(max_length):

            # generate new logit with control
            avg_direction = self.get_direction_from_state_list(pos_state_list, neg_state_list)

            temp_state = self._modify_state(copy.deepcopy(normal_state),
                                            avg_direction,
                                            control_coeff,
                                            lambda x: len(x.shape) != 1)
            logit, _ = self.forward(input_id, temp_state)

            # update states
            normal_state = self.forward(input_id, normal_state)[1]
            for idx in range(len(pos_state_list)):
                pos_state_list[idx] = self.forward(input_id, pos_state_list[idx])[1]
                neg_state_list[idx] = self.forward(input_id, neg_state_list[idx])[1]

            # sample new token
            input_id = self._top_kp_sampling(logit, temperature=temperature, k=top_k, p=top_p)

            if input_id == 0:
                break

            generated.append(input_id)

        return generated

    @torch.no_grad()
    def _implemented_run_loglikelihood(self, eval_args: EvalArgs) -> tuple[float, float]:

        pos_state_list = copy.deepcopy(self.pos_pre_state_list)
        neg_state_list = copy.deepcopy(self.neg_pre_state_list)

        normal_state = self.model.forward(eval_args.question_ids, None)[1]

        logits = []

        for token_pos in range(len(eval_args.answer_ids[:-1])):

            current_token = eval_args.answer_ids[token_pos]

            # add control
            avg_direction = self.get_direction_from_state_list(pos_state_list, neg_state_list)

            temp_state = self._modify_state(copy.deepcopy(normal_state),
                                            avg_direction,
                                            eval_args.control_coeff,
                                            layer_rule=lambda x: len(x.shape) != 1)

            logit, _ = self.forward(current_token, temp_state)

            logits.append(logit)

            # update states
            for idx in range(len(pos_state_list)):
                pos_state_list[idx] = self.forward(current_token, pos_state_list[idx])[1]
                neg_state_list[idx] = self.forward(current_token, neg_state_list[idx])[1]
            normal_state = self.forward(current_token, normal_state)[1]

        # Move the entire list of logits to the CPU if necessary, after the loop
        logits = torch.stack(logits).cpu()
        distributions = torch.softmax(logits, dim=-1)
        target_ids = eval_args.answer_ids[1:]

        # Use tensor operations to gather target probabilities
        target_probs = distributions[range(len(target_ids)), target_ids]

        loglikelihood = torch.log(target_probs)

        # Calculate the sum of the log likelihoods where the target mask is True, then normalize
        target_loglikelihood = loglikelihood.sum().item()

        # target_loglikelihood_norm = target_loglikelihood / (len(eval_args.answer_ids) - 1)
        target_loglikelihood_norm = target_loglikelihood / eval_args.answer_length

        return target_loglikelihood, target_loglikelihood_norm
