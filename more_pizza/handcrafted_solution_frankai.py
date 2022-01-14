"""
Handcrafted solution for Google's hash hackaton (https://codingcompetitions.withgoogle.com/hashcode) exercise "more_pizza"
Author: https://github.com/frank-ihle
"""

import csv


def get_dataset(filepath: str) -> dict:
    """
    Extracts the pizza environment
    """
    dataset = dict()
    with open(filepath) as file_reader:
        file_content = csv.reader(file_reader, delimiter=' ')

        for idl, line in enumerate(file_content):
            if idl == 0:
                dataset.update({"max_slices": int(line[0]), "pizza_variants": int(line[1])})
            elif idl == 1:
                sizes = [int(cur_size) for cur_size in line]
                sizes.sort()
                dataset.update({"sizes": sizes})

    return dataset

def find_optimum(dataset: dict, debug_print=False) -> list[int]:
    """
    Processes the dataset to find a suitable solution or an approximation.

    The idea is to start with the biggest slices and try to fill it with smaller slices until the target amount of slices was matched
    :param debug_print: whether to print intermediate results 
    :param dataset: pizza environment
    """
    def rough_approximation(dataset, last_max_variant, debug_print) -> (int, list[int]):
        """
        Looks for the closest approximation with the current max. pizza slice. This is an iterative process where the max. pizza slice is 
        continously decreased if the optimum was not found in one complete round (rough + fine approximation).
        :param last_max_variant: the current max. pizza slice to roughly approximate
        :param dataset: pizza environment
        :param debug_print: whether to print intermediate results 
        :return latest approximation, list like number of multiples of individual pizza slices
        """
        solution = [0 for _ in range(dataset["pizza_variants"])]
        cur_state = 0
        exceeded_target = False
        while not exceeded_target:
            if cur_state + dataset["sizes"][last_max_variant] <= dataset["max_slices"]:
                cur_state += dataset["sizes"][last_max_variant]
                solution[last_max_variant] += 1
            else:
                exceeded_target = True
        if debug_print:
            print("First approximation is {} with {} times slice-size {}".format(cur_state,
                                                                                 solution[last_max_variant],
                                                                                 dataset["sizes"][last_max_variant]))
        return cur_state, solution
    
    def fine_approximation(pizza_variant, cur_state, dataset, debug_print) -> (int, list[int]):
        """
        Tries to fill the remaining gab with the currently selected smaller pizza slice (pizza_variant) compared to the cur max. pizza slice.
        :param pizza_variant: the current smaller pizza slice to fill the gap to the target amount
        :param cur_state: latest state of approximation to the targtet amount (rough + fine approx)
        :param dataset: pizza environment
        :param debug_pring: whether to print intermediate results
        :return latest approximation, list like number of multiples of individual pizza slices
        """
        while pizza_variant >= 0 and cur_state < dataset["max_slices"]:
            exceeded_target = False
            while not exceeded_target:
                if cur_state + dataset["sizes"][pizza_variant] <= dataset["max_slices"]:
                    solution[pizza_variant] += 1
                    cur_state += dataset["sizes"][pizza_variant]
                else: 
                    exceeded_target = True
            if debug_print:
                print("- Cur. detail approx. '{}' with slice '{}' for the target '{}'".format(cur_state,
                                                                                              dataset["sizes"][pizza_variant],
                                                                                              dataset["max_slices"]))
            pizza_variant -= 1
        
        return cur_state, solution

    found_optimum = False
    last_max_variant = dataset["pizza_variants"]

    while not found_optimum:
        last_max_variant -= 1
        # get close to limit
        cur_state, solution = rough_approximation(dataset, last_max_variant, debug_print)

        # fill with remaining smaller slices
        pizza_variant = last_max_variant - 1
        cur_state, solution = fine_approximation(pizza_variant, cur_state, dataset, debug_print)


        if cur_state == dataset["max_slices"]:
            found_optimum = True
        elif last_max_variant == 0:
            break

    if debug_print:
        print("Final solution list '{}'".format(solution))
    return solution

def print_result(dataset: dict, solution: list):
    resultstr = ""
    approximation = dataset["max_slices"]
    for ids, cur_element in enumerate(solution):
        approximation -= dataset["sizes"][ids] * solution[ids]
        resultstr += "#{}:{}x \n".format(dataset["sizes"][ids], solution[ids])

    print("___________________________________________")
    print("Final result\n")
    print(resultstr)
    print("that's {} slices away from the target".format(approximation))

if __name__ == "__main__":

    # select dataset by commenting out subsequent dset_filepath definitions
    dset_filepath = "./data/a_example.in"
    dset_filepath = "./data/b_small.in"
    dset_filepath = "./data/c_medium.in"
    #dset_filepath = "./data/d_quite_big.in"
    #dset_filepath = "./data/e_also_big.in"

    dataset = get_dataset(dset_filepath)
    solution = find_optimum(dataset)
    print_result(dataset, solution)

