class dfs:
    def __init__(self, accepted_states: list,current_state: str,transitions: dict):
        self.accepted_states = accepted_states
        self.current_state = current_state
        self.transitions = transitions

    def transition(self, input_symbol: str):
        if self.current_state in self.transitions and input_symbol in self.transitions[self.current_state]:
            self.current_state = self.transitions[self.current_state][input_symbol]
            return True
        else:
            return False
        
    def rundfa(self, input_string: str):
         # Process the input string through the DFA
        for char in input_string:
            symbol = self.get_symbol(char)
            if not self.transition(symbol):
                print("Rejected: The input string is invalid.")
                break
        else:
            if self.current_state in self.accepted_states:
                print("Accepted: The input string has been processed successfully.")
            else:
                print("Rejected: The input string is invalid.")
    def get_symbol(self,char):

        if char.isalpha() and char.islower():
            return 'a'

        return 'b'
if __name__ == "__main__":
    # Define the DFA components
    accepted_states = ['q1']
    initial_state = 'q0'
    transitions = {
        'q0': {'a': 'q1', 'b': 'q2'},
        'q1': {'a': 'q1', 'b': 'q2'},
        'q2': {'a': 'q2', 'b': 'q2'}
    }

    # Create a DFA instance
    dfa = dfs(accepted_states, initial_state, transitions)

    # Get input string from user
    input_string = input("Enter a string: ")

    # Run the DFA on the input string
    dfa.rundfa(input_string)

   