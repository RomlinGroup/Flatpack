import random


def generate_command():
    command_type = random.choice(["rotateBase", "moveShoulder", "moveElbow"])
    if command_type == "rotateBase":
        value = random.uniform(-180, 180)
    elif command_type == "moveShoulder":
        value = random.uniform(0, 45)
    else:  # moveElbow
        value = random.uniform(-90, 45)
    return f"{command_type} {value:.2f}"


def generate_dataset(num_entries):
    return [generate_command() for _ in range(num_entries)]


dataset = generate_dataset(10000)

with open("fake_robot_dataset.txt", "w") as file:
    for command in dataset:
        file.write(command + "\n")

print("Dataset generated and saved as fake_robot_dataset.txt")
