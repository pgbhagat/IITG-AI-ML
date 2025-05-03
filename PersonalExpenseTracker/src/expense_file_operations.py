import csv
import os


def initialize_expense_file():
    """
        initialize the expense file
        create a new one if doesn't exist
    """
    expenses = []
    if not os.path.exists('expense.csv'):
        with open('expense.csv', 'w', newline='\n') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Date", "Amount", "Category", "Description"])
    else:
        expenses = load_expenses()
    return expenses


def save_expenses(expenses):
    """
        Save all expenses into the file,
        override it
    """
    with open('expense.csv', 'w', newline='\n') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Date", "Amount", "Category", "Description"])
        for expense in expenses:
            writer.writerow([expense['Date'], expense['Amount'], expense['Category'], expense['Description']])


def load_expenses():
    """
        Load all expenses from the file,
    """
    expenses = []
    with open('expense.csv', 'r', newline='\n') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # skip the header row
        for row in reader:
            expenses.append({'Date': row[0], 'Amount': row[1], 'Category': row[2], 'Description': row[3]})
    return expenses


def track_budget(month, budget):
    total = 0
    with open('expense.csv', 'r', newline='\n') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # skip the header
        for row in reader:
            if row[0].startswith(month):
                total += float(row[1])
    if total > budget:
        print('You have exceeded your budget!')
    else:
        print(f'You have Rs. {budget - total} left for this month')
