from roc.expmod import ExpMod


class MyTestExpModClass(ExpMod):
    modtype = "test"


@ExpMod.register("testy")
class MyTestExpMod(MyTestExpModClass):
    def do_thing(self) -> int:
        return 31337
