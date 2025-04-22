N = 8
def printSolution(board): 
    for row in board:
        print(" ".join("Q" if x else "." for x in row))
    print ("\n")
def isSafe(board, row, col): 
    for i in range(col):
        if board[row][i] == 1: 
            return False
    for i, j in zip(range(row, -1, -1), range(col, -1, -1)):
        if board[i][j] == 1:
            return False
    for i, j in zip(range(row, N, 1), range(col, -1, -1)):
        if board[i][j] == 1:
            return False

    return True
def solveNQueens(board, col):
    if col >= N:
        printSolution(board)
        return True

    res = False
    for row in range(N) :
        if isSafe(board, row, col):
            board[row][col] = 1
            res = solveNQueens (board, col + 1) or res
            board[row][col] = 0 # Backtrack

    return res

def nQueens():
    board = [[0] * N for _ in range(N)]
    if not solveNQueens(board, 0):
        print("No solution exists")

def main():
    nQueens()

if __name__ == "__main__": 
    main()
