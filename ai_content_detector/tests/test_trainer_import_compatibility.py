"""Historical trainer imports and serialized reward statistics remain usable."""

import pickle

import numpy as np

from ai_content_detector.rl_evasion.image_evasion import ddpo_trainer
from ai_content_detector.rl_evasion.image_evasion.ddpo_lifecycle import DDPOLifecycle
from ai_content_detector.rl_evasion.image_evasion.image_rewards import PerPromptStatTracker
from ai_content_detector.rl_evasion.text_evasion.grpo_optimization import GRPOOptimization
from ai_content_detector.rl_evasion.text_evasion.grpo_trainer import GRPOTextEvasionTrainer


def test_public_trainers_keep_their_original_paths_and_methods():
    assert GRPOTextEvasionTrainer.__module__.endswith("text_evasion.grpo_trainer")
    assert GRPOTextEvasionTrainer._train_with_trl is GRPOOptimization._train_with_trl
    assert ddpo_trainer.DDPOImageEvasionTrainer._evaluate is DDPOLifecycle._evaluate
    assert ddpo_trainer.PerPromptStatTracker is PerPromptStatTracker
    assert PerPromptStatTracker.__module__.endswith("image_evasion.ddpo_trainer")


def test_reward_tracker_pickle_keeps_history_and_next_update():
    original = PerPromptStatTracker(buffer_size=3)
    original.update(["one", "two", "one"], [0.1, 0.4, 0.2])
    restored = pickle.loads(pickle.dumps(original))
    assert restored._stats == original._stats
    np.testing.assert_array_equal(restored.update(["one", "two"], [0.7, 0.3]),
                                  original.update(["one", "two"], [0.7, 0.3]))
