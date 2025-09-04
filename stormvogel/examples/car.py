import stormvogel.model


def create_car_mdp():
    car = stormvogel.model.new_mdp()
    gs = car.get_initial_state()
    gs.add_label("green light")
    gs.add_label("still")
    rs = car.new_state(["red light", "still"])
    gm = car.new_state(["green light", "moving"])
    rm = car.new_state(["red light", "moving"])
    accident = car.new_state("accident")

    drive = car.new_action("accelerate")
    brake = car.new_action("brake")
    wait = car.new_action("wait")

    gs.set_choice([(drive, gm), (wait, rs)])
    rs.set_choice([(drive, rm), (wait, gs)])
    gm.set_choice([(brake, gs), (wait, rm)])
    rm.set_choice([(brake, rs), (wait, accident)])
    car.add_self_loops()
    return car
