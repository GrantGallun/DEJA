import json
import os

from smartreplay.frontend.instance import DesktopInstance


def test_second_launch_focuses_owner_instead_of_starting_another_app(tmp_path):
    focused = []
    with DesktopInstance(tmp_path, focus=lambda pid: True) as first:
        assert first.primary
        with DesktopInstance(tmp_path, focus=lambda pid: focused.append(pid) or True) as second:
            assert not second.primary
        assert focused == [os.getpid()]
    with DesktopInstance(tmp_path) as after_close:
        assert after_close.primary


def test_stale_desktop_owner_file_does_not_block_new_launch(tmp_path):
    (tmp_path / "desktop-owner.json").write_text(json.dumps({"pid": 999999}))
    focused = []
    with DesktopInstance(tmp_path, focus=lambda pid: focused.append(pid)) as owner:
        assert owner.primary
        assert not focused


def test_pending_window_does_not_create_second_service(tmp_path):
    with DesktopInstance(tmp_path) as owner:
        with DesktopInstance(tmp_path, focus=lambda pid: False, wait_seconds=.02) as duplicate:
            assert not duplicate.primary
