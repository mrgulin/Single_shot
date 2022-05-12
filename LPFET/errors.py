from LPFET import essentials

class DegeneratedStatesError(Exception):
    """Exception raised when the system has degenerated ground state.

    Attributes:
        n_e -- Number of electrons
        epsilon_s -- energy levels
        custom_message -- custom message to overwrite error message
    """

    def __init__(self, n_e, epsilon_s, custom_message=''):
        if not custom_message:
            self.message = f"Problem: It looks like there are degenerated energy levels. {n_e}, {epsilon_s}"
        else:
            self.message = custom_message
        super().__init__(self.message)


class HouseholderTransformationError(Exception):
    """Exception raised when the Householder transformation wasn't successful.

    Attributes:
        one_rdm -- 1RDM that was input for the Householder transformation
        message -- custom message to overwrite error message
    """

    def __init__(self, one_rdm, custom_message=''):
        if not custom_message:
            self.message = f"Error: couldn't create Householder transformation for a given 1RDM. 1RDM:\n"
            self.message += essentials.print_matrix(one_rdm)
        else:
            self.message = custom_message
        super().__init__(self.message)


class EmptyFullOrbitalError(Exception):
    """Exception raised when KS density is either 0 or 2.

    Attributes:
        n_ks -- vector of densities
        value -- wrong value
        message -- custom message to overwrite error message
    """

    def __init__(self, n_ks, value, custom_message=''):
        if not custom_message:
            self.message = f"Error: KS density son some site is {value} which is too close {round(value[0]):.0f}" \
                           f" (densities: {n_ks})\n"
        else:
            self.message = custom_message
        super().__init__(self.message)
