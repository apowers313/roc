# # mypy: disable-error-code="no-untyped-def"

# from nle.env.base import FULL_ACTIONS

# from roc.action import Action, ActionCount


# class TestAction:
#     def test_action_count(self, action_bus_conn, mocker) -> None:
#         ad = ActionCount(action_count=42)
#         print("Creating action")
#         ac = Action()
#         print("Action created")
#         assert ac.action_count is None
#         mocker.spy(Action, "recv_action_count")
#         action_bus_conn.send(ad)

#         # spy.assert_called_once()
#         # assert spy.call_count == 1
#         assert ac.action_count == 42

#     def test_delete_me(self) -> None:
#         # print(FULL_ACTIONS)
#         # print(FULL_ACTIONS[19])
#         # print(chr(FULL_ACTIONS[19]))
#         # print(FULL_ACTIONS[0])
#         # print(chr(FULL_ACTIONS[0]))
#         for a in FULL_ACTIONS:
#             la = list(FULL_ACTIONS)
#             idx = la.index(a)
#             print(idx, a.name)
