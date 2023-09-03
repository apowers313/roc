import pprint
from typing import Any

import click
import nle  # noqa

import roc

pp = pprint.PrettyPrinter(width=41, compact=True)


def ascii_list(al: list[int]) -> str:
    result_string = ""

    for ascii_value in al:
        result_string += chr(ascii_value)

    return result_string


def int_list(al: list[int]) -> str:
    result_string = ""

    for ascii_value in al:
        result_string += str(ascii_value) + " "

    return result_string


def print_screen(screen: list[list[int]], *, as_int: bool = False) -> None:
    for row in screen:
        if not as_int:
            print(ascii_list(row))
        else:
            print(int_list(row))


# class Environment(Component):
#     def __init__(self) -> None:
#         super().__init__("environment", "environment")
#         # self.attach(perception_bus)
#         perception_bus


@click.command
@click.option("--arg", default=1)
def cli(arg: Any) -> None:
    # env = gym.make("NetHackScore-v0", options=["autodig"])
    # env = gym.make("NetHackScore-v0")
    # nhgym = NethackGym(env)
    # nhgym.start()
    # exit()
    roc.init()
    roc.start()

    # # GymComponent(gym.make("NetHackScore-v0"))
    # # environment_bus.send(EnvInputEvent(env.action_space))
    # print(repr(env.action_space))
    # print(env.action_space)
    # print(env.action_space.shape)
    # print(env.action_space.dtype)
    # print(env.action_space.n)
    # # from gym.spaces.discrete import Discrete

    # # d = Discrete(3, start=-1)
    # # print(d.n)
    # # print(0 in d)
    # # print(-1 in d)

    # # print(repr(env.observation_space))
    # # print(repr(env.reward_range))
    # # print(repr(env.action_space.sample()))
    # # print(repr(env.action_space.sample()))
    # # print(repr(env.action_space.sample()))
    # env.print_action_meanings()
    # (obs) = env.reset()
    # pp.pprint(obs)
    # print(ascii_list(obs["message"]))
    # print("observation keys:", obs.keys())
    # print(obs["blstats"])

    # # options https://nethackwiki.com/wiki/Options
    # print("default options", nethack.NETHACKOPTIONS)
    # # default options ('autopickup', 'color', 'disclose:+i +a +v +g +c +o', 'mention_walls', 'nobones', 'nocmdassist', 'nolegacy', 'nosparkle', 'pickup_burden:unencumbered', 'pickup_types:$?!/', 'runmode:teleport', 'showexp', 'showscore', 'time') # noqa
    # print("nethack options", env.nethack.options)
    # # NETHACKOPTIONS=boulder:0, color, autodig
    # # gym.make(options=["boulder:0"])

    # # bottom line
    # print(obs["blstats"][nethack.NLE_BL_SCORE])
    # print("hp", obs["blstats"][nethack.NLE_BL_HP])
    # print("max hp", obs["blstats"][nethack.NLE_BL_HPMAX])
    # # /* blstats indices, see also botl.c and statusfields in botl.h. */
    # # define NLE_BL_X 0
    # # define NLE_BL_Y 1
    # # define NLE_BL_STR25 2  /* strength 3..25 */
    # # define NLE_BL_STR125 3 /* strength 3..125   */
    # # define NLE_BL_DEX 4
    # # define NLE_BL_CON 5
    # # define NLE_BL_INT 6
    # # define NLE_BL_WIS 7
    # # define NLE_BL_CHA 8
    # # define NLE_BL_SCORE 9
    # # define NLE_BL_HP 10
    # # define NLE_BL_HPMAX 11
    # # define NLE_BL_DEPTH 12
    # # define NLE_BL_GOLD 13
    # # define NLE_BL_ENE 14
    # # define NLE_BL_ENEMAX 15
    # # define NLE_BL_AC 16
    # # define NLE_BL_HD 17  /* monster level, "hit-dice" */
    # # define NLE_BL_XP 18  /* experience level */
    # # define NLE_BL_EXP 19 /* experience points */
    # # define NLE_BL_TIME 20
    # # define NLE_BL_HUNGER 21 /* hunger state */
    # # define NLE_BL_CAP 22    /* carrying capacity */
    # # define NLE_BL_DNUM 23
    # # define NLE_BL_DLEVEL 24
    # # define NLE_BL_CONDITION 25 /* condition bit mask */
    # # define NLE_BL_ALIGN 26

    # print("condition", obs["blstats"][nethack.NLE_BL_CONDITION])
    # print(nethack.BL_MASK_BITS)
    # print(nethack.BL_MASK_BLIND)
    # print(dir(nethack))
    # # /* Boolean condition bits for the condition mask */
    # # define BL_MASK_BAREH        0x00000001L
    # # define BL_MASK_BLIND        0x00000002L
    # # define BL_MASK_BUSY         0x00000004L
    # # define BL_MASK_CONF         0x00000008L
    # # define BL_MASK_DEAF         0x00000010L
    # # define BL_MASK_ELF_IRON     0x00000020L
    # # define BL_MASK_FLY          0x00000040L
    # # define BL_MASK_FOODPOIS     0x00000080L
    # # define BL_MASK_GLOWHANDS    0x00000100L
    # # define BL_MASK_GRAB         0x00000200L
    # # define BL_MASK_HALLU        0x00000400L
    # # define BL_MASK_HELD         0x00000800L
    # # define BL_MASK_ICY          0x00001000L
    # # define BL_MASK_INLAVA       0x00002000L
    # # define BL_MASK_LEV          0x00004000L
    # # define BL_MASK_PARLYZ       0x00008000L
    # # define BL_MASK_RIDE         0x00010000L
    # # define BL_MASK_SLEEPING     0x00020000L
    # # define BL_MASK_SLIME        0x00040000L
    # # define BL_MASK_SLIPPERY     0x00080000L
    # # define BL_MASK_STONE        0x00100000L
    # # define BL_MASK_STRNGL       0x00200000L
    # # define BL_MASK_STUN         0x00400000L
    # # define BL_MASK_SUBMERGED    0x00800000L
    # # define BL_MASK_TERMILL      0x01000000L
    # # define BL_MASK_TETHERED     0x02000000L
    # # define BL_MASK_TRAPPED      0x04000000L
    # # define BL_MASK_UNCONSC      0x08000000L
    # # define BL_MASK_WOUNDEDL     0x10000000L
    # # define BL_MASK_HOLDING      0x20000000L
    # # define BL_MASK_BITS            30 /* number of mask bits that can be set */

    # # # print(type(obs["blstats"]))
    # # # print(obs["blstats"].shape)
    # # ic(obs["tty_chars"])
    # # print_screen(obs["tty_chars"])
    # # ic(obs["screen_descriptions"])
    # # print(type(obs["screen_descriptions"]))
    # # print(obs["screen_descriptions"].shape)
    # # # for screen in obs["screen_descriptions"]:
    # # #     print("next screen")
    # # #     for row in screen:
    # # #         print(ascii_list(row))
    # # # env.render()

    # # # obs, reward, terminated, truncated = env.step(env.action_space.sample())
    # # # pp.pprint(obs)
    # # # pp.pprint(reward)
    # # # print("terminated:", terminated)
    # # # print("truncated:", truncated)
    # # print_screen(obs["chars"])
    # # print_screen(obs["colors"], as_int=True)
    # # print_screen(obs["inv_strs"])
    # # print(obs["glyphs"])
    # # print(type(obs["glyphs"]))
    # # print(obs["inv_oclasses"].shape)
    # # print(obs["inv_oclasses"])


if __name__ == "__main__":
    cli()
