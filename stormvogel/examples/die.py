import stormvogel.model
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont


def create_die_dtmc():
    # Create a new model with the name "Die"
    dtmc = stormvogel.model.new_dtmc()

    init = dtmc.get_initial_state()
    init.add_valuation("rolled", 0)

    # From the initial state, add the transition to 6 new states with probability 1/6th.
    init.set_choice(
        [
            (1 / 6, dtmc.new_state(labels=f"rolled{i+1}", valuations={"rolled": i + 1}))
            for i in range(6)
        ]
    )

    # we add self loops to all states with no outgoing choices
    dtmc.add_self_loops()

    # test if state removal works
    # dtmc.remove_state(dtmc.get_state_by_id(1), True, True)

    return dtmc


def generate_dice_image(number: int) -> Image.Image:
    """
    Generates a 128x128 PIL image with a question mark for 0
    or a dice face (1-6) with the corresponding number of dots.
    """
    img = Image.new("RGB", (128, 128), "white")
    draw = ImageDraw.Draw(img)

    # Draw border
    draw.rectangle([(5, 5), (123, 123)], outline="black", width=5)  # type: ignore

    if number == 0:
        draw.text((40, 24), "?", font=ImageFont.load_default(80), fill="black")
        return img

    # Dice dot positions relative to a 128x128 grid
    positions = {
        1: [(64, 64)],
        2: [(40, 40), (88, 88)],
        3: [(40, 40), (64, 64), (88, 88)],
        4: [(40, 40), (40, 88), (88, 40), (88, 88)],
        5: [(40, 40), (40, 88), (88, 40), (88, 88), (64, 64)],
        6: [(32, 32), (32, 64), (32, 96), (96, 32), (96, 64), (96, 96)],
    }

    for pos in positions.get(number, []):
        draw.ellipse(
            [(pos[0] - 12, pos[1] - 12), (pos[0] + 12, pos[1] + 12)], fill="black"
        )

    return img


if __name__ == "__main__":
    # Print the resulting model in dot format.

    print(create_die_dtmc().to_dot())
