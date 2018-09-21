class Car:
    def __init__(self):
        self.speed = 0
        self.odometer = 0
        self.time = 0

    def say_state(self):
        print("I'm going {} kph!".format(self.speed))
        print("Time taken = {}" .format(self.time))

    def accelerate(self):
        self.speed += 5

    def brake(self):
        self.speed -= 5

    def step(self):
        self.odometer += self.speed
        self.time += 1

    def average_speed(self):
        if self.time == 0:
            return self.odometer / 1
        else:
            return self.odometer / self.time


if __name__ == '__main__':
    my_car = Car()
    print("I'm a car!")
    while True:
        action = input("What should I do? [a]ccelerate, [b]rake, "
            "show [o]dometer, or show average [s]peed?\n$ ").lower()

        if action not in "abos" or len(action) != 1:
            print("I don't know how to do that")
            continue

        if action == 'a':
            my_car.accelerate()
        elif action == 'b':
            my_car.brake()
        elif action == 'o':
            print("The car has driven {} kilometers".format(my_car.odometer))
        elif action == 's':
            print("The car's average speed was {} kph".format(my_car.average_speed()))

        my_car.step()
        my_car.say_state()