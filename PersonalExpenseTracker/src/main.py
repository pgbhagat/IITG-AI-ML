import datetime
import expense_file_operations as exp_file


def main():
    expenses = exp_file.initialize_expense_file()
    while True:
        print()
        print('1. Add Expense')
        print('2. View Expenses')
        print('3. Track Budget')
        print('4. Save Expenses')
        print('5. Exit')
        print()
        choice = input('Enter your choice: ')
        if choice == '1':
            date = input('Enter the date(yyyy-mm-dd): ')
            if not validate_date(date):
                print('Invalid date')
                continue
            amount = input('Enter the amount in Rs: ')
            if not validate_amount(amount):
                print('Invalid amount')
                continue
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
            if not validate_date(month, '%Y-%m'):
                print('Invalid date')
                continue

            budget = input(f'Enter the total budget amount Rs for the month {month}: ')
            if not validate_amount(budget):
                print('Invalid budget amount')
                continue
            exp_file.save_expenses(expenses)
            exp_file.track_budget(month, float(budget))
        elif choice == '4':
            exp_file.save_expenses(expenses)
        elif choice == '5':
            break
        else:
            print('Invalid choice\n')


def validate_date(date, format='%Y-%m-%d'):
    try:
        datetime.datetime.strptime(date, format)
        return True
    except ValueError:
        return False


def validate_amount(amount):
    try:
        a = float(amount)
        if a <= 0:
            raise ValueError('Amount must be positive')
        return True
    except ValueError:
        return False


if __name__ == '__main__':
    main()
