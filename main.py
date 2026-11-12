import ex1, ex2, ex3

if __name__ == "__main__":
    while True:
        choice = input("Enter an exercise and question (e.g. 2 d) / type the end, q or quit : ").lower()

        if choice == "end" or choice == "q" or choice == "quit":
            print("END")
            break
        elif choice == "1 a":
            print("Solve in a notebook")
        elif choice == "1 b":
            print("Check the Function.")
        elif choice == "1 c":
            ex1.c()
        elif choice == "1 d":
            ex1.d()
        elif choice == "1 e":
            ex1.e()
        elif choice == "2 a":
            ex2.a()
        elif choice == "2 b":
            ex2.b()
        elif choice == "2 c":
            ex2.c()
        elif choice == "2 d":
            ex2.d()
        elif choice == "3 c":
            ex3.c()
        elif choice == "3 d":
            ex3.d()
        elif choice == "3 e":
            ex3.e()
        elif choice == "3 f":
            ex3.f()
        else:
            print("Error! ex1: b~e, ex2: a~d, ex3: c~f")
