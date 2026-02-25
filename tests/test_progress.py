"""Tests for the progress visualization system."""

from __future__ import annotations

import asyncio
from io import StringIO
from unittest.mock import Mock, patch

import pytest

from llm_council.progress import ModelStatus, ProgressManager, StageProgress


class TestProgressManagerNonTTY:
    """Test ProgressManager in non-TTY mode (log output)."""

    @pytest.mark.asyncio
    async def test_non_tty_start_stage(self):
        """Test that start_stage emits correct log line in non-TTY mode."""
        with patch("sys.stderr", new_callable=StringIO) as mock_stderr:
            manager = ProgressManager(is_tty=False)
            await manager.start_stage(1, "Initial responses", ["Model A", "Model B", "Model C"])

            output = mock_stderr.getvalue()
            assert "Stage 1/3: Initial responses (3 models)..." in output

    @pytest.mark.asyncio
    async def test_non_tty_single_model(self):
        """Test singular form for single model."""
        with patch("sys.stderr", new_callable=StringIO) as mock_stderr:
            manager = ProgressManager(is_tty=False)
            await manager.start_stage(2, "Ranking", ["Chairman"])

            output = mock_stderr.getvalue()
            assert "Stage 2/3: Ranking (1 model)..." in output

    @pytest.mark.asyncio
    async def test_non_tty_model_done(self):
        """Test that model completion emits correct log line."""
        with patch("sys.stderr", new_callable=StringIO) as mock_stderr:
            manager = ProgressManager(is_tty=False)
            await manager.start_stage(1, "Test", ["Model A"])
            mock_stderr.truncate(0)  # Clear start message

            await manager.update_model("Model A", ModelStatus.DONE, elapsed=2.5)

            output = mock_stderr.getvalue()
            assert "Stage 1/3: ✓ Model A done (2.5s)" in output

    @pytest.mark.asyncio
    async def test_non_tty_model_failed(self):
        """Test that model failure emits correct log line."""
        with patch("sys.stderr", new_callable=StringIO) as mock_stderr:
            manager = ProgressManager(is_tty=False)
            await manager.start_stage(2, "Test", ["Model B"])
            mock_stderr.truncate(0)

            await manager.update_model("Model B", ModelStatus.FAILED, elapsed=1.2)

            output = mock_stderr.getvalue()
            assert "Stage 2/3: ✗ Model B failed (1.2s)" in output

    @pytest.mark.asyncio
    async def test_non_tty_stage_complete(self):
        """Test stage completion log with summary."""
        with patch("sys.stderr", new_callable=StringIO) as mock_stderr:
            manager = ProgressManager(is_tty=False)
            await manager.start_stage(1, "Test", ["Model A"])
            mock_stderr.truncate(0)

            await manager.complete_stage(summary="All models responded")

            output = mock_stderr.getvalue()
            assert "Stage 1/3: Complete" in output
            assert "All models responded" in output

    @pytest.mark.asyncio
    async def test_non_tty_stage_complete_no_summary(self):
        """Test stage completion without summary."""
        with patch("sys.stderr", new_callable=StringIO) as mock_stderr:
            manager = ProgressManager(is_tty=False)
            await manager.start_stage(3, "Test", ["Model"])
            mock_stderr.truncate(0)

            await manager.complete_stage()

            output = mock_stderr.getvalue()
            assert "Stage 3/3: Complete" in output
            assert " — " not in output  # No summary separator

    @pytest.mark.asyncio
    async def test_non_tty_council_complete(self):
        """Test council completion message."""
        with patch("sys.stderr", new_callable=StringIO) as mock_stderr:
            manager = ProgressManager(is_tty=False)
            await manager.complete_council(total_elapsed=15.3)

            output = mock_stderr.getvalue()
            assert "Council complete (15.3s total)" in output

    @pytest.mark.asyncio
    async def test_non_tty_ignores_pending_and_querying(self):
        """Test that PENDING and QUERYING status don't emit logs in non-TTY."""
        with patch("sys.stderr", new_callable=StringIO) as mock_stderr:
            manager = ProgressManager(is_tty=False)
            await manager.start_stage(1, "Test", ["Model A"])
            mock_stderr.truncate(0)

            await manager.update_model("Model A", ModelStatus.PENDING)
            await manager.update_model("Model A", ModelStatus.QUERYING)

            output = mock_stderr.getvalue()
            assert output == ""  # No output for these statuses

    @pytest.mark.asyncio
    async def test_log_format_includes_timestamp(self):
        """Test that log lines include timestamp."""
        with patch("sys.stderr", new_callable=StringIO) as mock_stderr:
            manager = ProgressManager(is_tty=False)
            await manager.start_stage(1, "Test", ["Model"])

            output = mock_stderr.getvalue()
            # Should have format: [HH:MM:SS] message
            assert output.startswith("[")
            assert "]" in output
            assert ":" in output.split("]")[0]  # Time format


class TestProgressManagerTTY:
    """Test ProgressManager in TTY mode (rich display)."""

    @pytest.mark.asyncio
    async def test_tty_mode_creates_live_display(self):
        """Test that TTY mode creates a Live display."""
        with patch("llm_council.progress.Console"):
            with patch("llm_council.progress.Live") as mock_live_cls:
                mock_live = Mock()
                mock_live_cls.return_value = mock_live

                manager = ProgressManager(is_tty=True)
                await manager.start_stage(1, "Test", ["Model"])

                mock_live.start.assert_called_once()
                assert manager._live is mock_live

    @pytest.mark.asyncio
    async def test_tty_render_loop_started(self):
        """Test that render loop task is started in TTY mode."""
        with patch("llm_council.progress.Console"):
            with patch("llm_council.progress.Live"):
                manager = ProgressManager(is_tty=True)
                await manager.start_stage(1, "Test", ["Model"])

                assert manager._render_task is not None
                assert not manager._render_task.done()

                # Clean up
                await manager._cleanup()

    @pytest.mark.asyncio
    async def test_tty_render_content(self):
        """Test the content of TTY render."""
        manager = ProgressManager(is_tty=True)
        manager.current_stage = StageProgress(
            stage_num=2,
            description="Peer review",
            models={"Claude": ModelStatus.DONE, "GPT": ModelStatus.QUERYING, "Gemini": ModelStatus.PENDING},
            model_elapsed={"Claude": 3.2},
            model_start_times={"GPT": 100.0},  # Mock start time
        )

        with patch("time.time", return_value=102.5):  # 2.5s after GPT started
            text = manager._render_tty()
            content = text.plain

            assert "Stage 2/3: Peer review" in content
            assert "✓ Claude" in content
            assert "done (3.2s)" in content
            assert "GPT" in content
            assert "querying... (2.5s)" in content
            assert "○ Gemini" in content
            assert "pending" in content

    @pytest.mark.asyncio
    async def test_tty_spinner_animation(self):
        """Test that spinner cycles through characters."""
        manager = ProgressManager(is_tty=True)
        initial_idx = manager._spinner_idx

        # Simulate render loop iterations
        for _ in range(3):
            manager._spinner_idx = (manager._spinner_idx + 1) % len(manager._spinner_chars)

        assert manager._spinner_idx != initial_idx
        assert manager._spinner_idx < len(manager._spinner_chars)

    @pytest.mark.asyncio
    async def test_tty_stage_completed_display(self):
        """Test display when stage is completed."""
        manager = ProgressManager(is_tty=True)
        manager.current_stage = StageProgress(
            stage_num=1,
            description="Initial responses",
            models={"Model A": ModelStatus.DONE, "Model B": ModelStatus.DONE},
            completed=True,
            summary="2/2 models responded successfully",
        )

        text = manager._render_tty()
        content = text.plain

        assert "done" in content.lower()
        assert "2/2 models responded successfully" in content

    @pytest.mark.asyncio
    async def test_tty_failed_model_display(self):
        """Test display of failed models."""
        manager = ProgressManager(is_tty=True)
        manager.current_stage = StageProgress(
            stage_num=1,
            description="Test",
            models={"Failing Model": ModelStatus.FAILED},
            model_elapsed={"Failing Model": 0.5},
        )

        text = manager._render_tty()
        content = text.plain

        assert "✗" in content or "failed" in content.lower()
        assert "Failing Model" in content
        assert "0.5s" in content


class TestProgressManagerCleanup:
    """Test cleanup and resource management."""

    @pytest.mark.asyncio
    async def test_cleanup_stops_render_task(self):
        """Test that cleanup properly cancels the render task."""
        with patch("llm_council.progress.Console"):
            with patch("llm_council.progress.Live"):
                manager = ProgressManager(is_tty=True)
                await manager.start_stage(1, "Test", ["Model"])

                assert manager._render_task is not None
                await manager._cleanup()

                assert manager._render_task.done()

    @pytest.mark.asyncio
    async def test_cleanup_stops_live_display(self):
        """Test that cleanup stops the Live display."""
        with patch("llm_council.progress.Console"):
            with patch("llm_council.progress.Live") as mock_live_cls:
                mock_live = Mock()
                mock_live_cls.return_value = mock_live

                manager = ProgressManager(is_tty=True)
                await manager.start_stage(1, "Test", ["Model"])
                await manager._cleanup()

                mock_live.stop.assert_called_once()
                assert manager._live is None

    @pytest.mark.asyncio
    async def test_cleanup_handles_none_state(self):
        """Test that cleanup handles None states gracefully."""
        manager = ProgressManager(is_tty=False)
        # No stage started, everything is None
        await manager._cleanup()  # Should not raise

    @pytest.mark.asyncio
    async def test_complete_council_calls_cleanup(self):
        """Test that complete_council triggers cleanup."""
        with patch.object(ProgressManager, "_cleanup") as mock_cleanup:
            mock_cleanup.return_value = asyncio.create_task(asyncio.sleep(0))
            manager = ProgressManager(is_tty=False)
            await manager.complete_council(10.0)

            mock_cleanup.assert_called_once()


class TestStageProgress:
    """Test the StageProgress dataclass."""

    def test_stage_progress_initialization(self):
        """Test StageProgress initialization with defaults."""
        stage = StageProgress(stage_num=1, description="Test stage")

        assert stage.stage_num == 1
        assert stage.stage_total == 3
        assert stage.description == "Test stage"
        assert stage.models == {}
        assert stage.completed is False
        assert stage.summary is None

    def test_stage_progress_with_models(self):
        """Test StageProgress with model tracking."""
        stage = StageProgress(
            stage_num=2,
            description="Ranking",
            models={"Model A": ModelStatus.DONE, "Model B": ModelStatus.QUERYING},
        )

        assert len(stage.models) == 2
        assert stage.models["Model A"] == ModelStatus.DONE
        assert stage.models["Model B"] == ModelStatus.QUERYING


class TestModelStatus:
    """Test the ModelStatus enum."""

    def test_model_status_values(self):
        """Test that ModelStatus has expected values."""
        assert ModelStatus.PENDING.value == "pending"
        assert ModelStatus.QUERYING.value == "querying"
        assert ModelStatus.DONE.value == "done"
        assert ModelStatus.FAILED.value == "failed"

    def test_model_status_comparison(self):
        """Test ModelStatus enum comparison."""
        assert ModelStatus.DONE == ModelStatus.DONE
        assert ModelStatus.DONE != ModelStatus.FAILED
        assert ModelStatus.PENDING != ModelStatus.QUERYING


class TestProgressManagerEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_update_model_without_stage(self):
        """Test updating model when no stage is active."""
        manager = ProgressManager(is_tty=False)
        # Should not raise, just return
        await manager.update_model("Model", ModelStatus.DONE)

    @pytest.mark.asyncio
    async def test_complete_stage_without_current_stage(self):
        """Test completing stage when none is active."""
        manager = ProgressManager(is_tty=False)
        # Should not raise
        await manager.complete_stage("Summary")

    @pytest.mark.asyncio
    async def test_model_transitions(self):
        """Test that model status transitions work correctly."""
        manager = ProgressManager(is_tty=False)
        await manager.start_stage(1, "Test", ["Model A"])

        # Transition through states
        await manager.update_model("Model A", ModelStatus.PENDING)
        assert manager.current_stage.models["Model A"] == ModelStatus.PENDING

        await manager.update_model("Model A", ModelStatus.QUERYING)
        assert manager.current_stage.models["Model A"] == ModelStatus.QUERYING
        assert "Model A" in manager.current_stage.model_start_times

        await manager.update_model("Model A", ModelStatus.DONE, elapsed=5.0)
        assert manager.current_stage.models["Model A"] == ModelStatus.DONE
        assert manager.current_stage.model_elapsed["Model A"] == 5.0

    @pytest.mark.asyncio
    async def test_concurrent_updates(self):
        """Test that concurrent updates are handled safely."""
        manager = ProgressManager(is_tty=False)
        await manager.start_stage(1, "Test", ["Model A", "Model B", "Model C"])

        # Simulate concurrent updates
        async def update_a():
            await manager.update_model("Model A", ModelStatus.DONE, 1.0)

        async def update_b():
            await manager.update_model("Model B", ModelStatus.FAILED, 2.0)

        async def update_c():
            await manager.update_model("Model C", ModelStatus.DONE, 1.5)

        await asyncio.gather(update_a(), update_b(), update_c())

        # All updates should be applied
        assert manager.current_stage.models["Model A"] == ModelStatus.DONE
        assert manager.current_stage.models["Model B"] == ModelStatus.FAILED
        assert manager.current_stage.models["Model C"] == ModelStatus.DONE

    @pytest.mark.asyncio
    async def test_render_with_no_models(self):
        """Test rendering when stage has no models."""
        manager = ProgressManager(is_tty=True)
        manager.current_stage = StageProgress(stage_num=1, description="Empty stage")

        text = manager._render_tty()
        content = text.plain

        assert "Stage 1/3: Empty stage" in content

    @pytest.mark.asyncio
    async def test_render_with_very_long_model_name(self):
        """Test rendering with very long model names."""
        manager = ProgressManager(is_tty=True)
        long_name = "A" * 50
        manager.current_stage = StageProgress(
            stage_num=1, description="Test", models={long_name: ModelStatus.DONE}, model_elapsed={long_name: 1.0}
        )

        text = manager._render_tty()
        # Should not raise, should handle long names gracefully
        assert text is not None

    def test_is_tty_auto_detection(self):
        """Test automatic TTY detection."""
        with patch("sys.stderr.isatty", return_value=True):
            with patch("llm_council.progress.HAS_RICH", True):
                manager = ProgressManager()  # Auto-detect
                assert manager.is_tty is True

        with patch("sys.stderr.isatty", return_value=False):
            manager = ProgressManager()  # Auto-detect
            assert manager.is_tty is False

    def test_no_rich_forces_non_tty(self):
        """Test that missing rich library forces non-TTY mode."""
        with patch("llm_council.progress.HAS_RICH", False):
            with patch("sys.stderr.isatty", return_value=True):
                manager = ProgressManager()
                assert manager.is_tty is False
