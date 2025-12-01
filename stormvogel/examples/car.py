import stormvogel.model


def create_car_mdp():
    car = stormvogel.model.new_mdp()
    gs = car.initial_state
    gs.add_label("green light")
    gs.add_label("still")
    rs = car.new_state(["red light", "still"])
    gm = car.new_state(["green light", "moving"])
    rm = car.new_state(["red light", "moving"])
    accident = car.new_state("accident")

    drive = car.new_action("accelerate")
    brake = car.new_action("brake")
    wait = car.new_action("wait")

    gs.set_choices([(drive, gm), (wait, rs)])
    rs.set_choices([(drive, rm), (wait, gs)])
    gm.set_choices([(brake, gs), (wait, rm)])
    rm.set_choices([(brake, rs), (wait, accident)])
    car.add_self_loops()
    return car
