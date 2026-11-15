import ex1, ex2, ex3

if __name__ == "__main__":
    while True:
        choice = input("Enter an exercise and question (e.g. 2 d) / type the end, q or quit : ").lower()

        if choice in ("q", "quit", "end"):
            break
        elif choice == "1 a":
            print("Solve in a notebook")
        elif choice == "1 b":
            print("Check the Function")
        elif choice == "1 c":
            ex1.c()
        elif choice == "1 d":
            ex1.d()
        elif choice == "1 e":
            ex1.e()
        elif choice == "2":
            ex2.show_results()
        elif choice == "2 a":
            print("inpuy the '2'.")
        elif choice == "2 b":
            print("inpuy the '2'.")
        elif choice == "2 c":
            print("inpuy the '2'.")
        elif choice == "3 a":
            ex3.a()
        elif choice == "3 b":
            ex3.b()
        elif choice == "3 c":
            print("Check the Function.")
        elif choice == "3 d":
            ex3.d()
        elif choice == "3 e":
            ex3.e()
        elif choice == "3 f":
            ex3.f()
        elif choice == "3 g":
            ex3.g()
        elif choice == "3 h":
            ex3.h()
        else:
            print("Error.")