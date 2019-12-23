class NoShipsException(Exception):
    def __init__(self):
        print('No ships on that picture! Choose another one')