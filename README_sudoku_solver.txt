Sudoku File Solver — How to use
================================

1) Upload your puzzle here in one of these formats (blanks can be '.' or '0'):
   - 81-character single line, e.g. 530070000600195000...
   - 9 lines of 9 symbols (spaces or commas allowed)
   - 9x9 CSV

2) Then run the CLI locally (optional):
   - Save the attached script: sudoku_solver.py
   - Usage:
       python3 sudoku_solver.py YOUR_PUZZLE.txt
     or pipe:
       cat YOUR_PUZZLE.txt | python3 sudoku_solver.py

3) What you’ll get:
   - A solved 9x9 grid printed to stdout
   - In this workspace, I can also display the solution as a table and save it to a file by name.

Files in this message:
 - /mnt/data/sudoku_solver.py
 - /mnt/data/sample_puzzle.txt
 - /mnt/data/solved_sample.txt
