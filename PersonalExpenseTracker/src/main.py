import datetime
import expense_file_operations as exp_file


def main():
    expenses = exp_file.initialize_expense_file()
    while True:
        print('1. Add Expense')
        print('2. View Expenses')
        print('3. Track Budget')
        print('4. Save Expenses')
        print('5. Exit')
        print()
        choice = input('Enter your choice: ')
        if choice == '1':
            date = input('Enter the date(yyyy-mm-dd): ')
            amount = input('Enter the amount in Rs: ')
            category = input('Enter the category: ')
            description = input('Enter the description: ')
            expenses.append({'Date': date, 'Amount': amount, 'Category': category, 'Description': description})
        elif choice == '2':
            if len(expenses) == 0:
                print('No expenses found')
            else:
                print('Expenses as below:')
                print('Date, Amount, Category, Description')
                for expense in expenses:
                    print(f'{expense['Date']}, {expense['Amount']}, {expense['Category']}, {expense['Description']}')
        elif choice == '3':
            month = input('Enter the month(yyyy-mm): ')
            budget = input(f'Enter the total budget amount Rs for the month {month}: ')
            exp_file.save_expenses(expenses)
            exp_file.track_budget(month, float(budget))
        elif choice == '4':
            exp_file.save_expenses(expenses)
        elif choice == 5:
            break


if __name__ == '__main__':
    main()
