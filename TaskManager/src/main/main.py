from src.user.user_manager import UserManager
from src.task.task_manager import TaskManager


def main():
    user_manager = UserManager()
    task_manager = TaskManager()
    logged_in_user = ''

    while True:
        print('--Task manager--')
        print('1. Register')
        print('2. Login')
        print('3. Exit')
        choice = input('Enter your choice: ')

        if choice == '1':
            username = input('Choose a username: ')
            password = input('Choose a password: ')
            repassword = input('Reenter the password: ')
            if password != repassword:
                print('Passwords do not match')
                continue
            _, message = user_manager.register_new_user(username, password)
            print(message)

        elif choice == '2':
            username = input('Enter your username: ')
            password = input('Enter the password: ')
            ret, message = user_manager.login(username, password)
            print(message)

            if ret:
                logged_in_user = username
                while True:
                    print('1. Add a task')
                    print('2. View all tasks')
                    print('3. Delete a task')
                    print('4. Complete a task')
                    print('5. Logout')
                    choice1 = input('Enter your choice: ')
                    if choice1 == '1':
                        task_description = input('Add a task, task description: ')
                        task_manager.add_task(logged_in_user, task_description)
                    elif choice1 == '2':
                        print('All tasks -- ')
                        tasks = task_manager.view_task(logged_in_user)
                        if tasks is not None:
                            for task in tasks:
                                print(task)
                        else:
                            print('No tasks')
                    elif choice1 == '3':
                        task_id = input('Delete a task, task id: ')
                        if task_id.isdigit():
                            task_manager.delete_task(logged_in_user, int(task_id))
                        else:
                            print('Invalid task id')

                    elif choice1 == '4':
                        task_id = input('Complete a task, task id: ')
                        if task_id.isdigit():
                            task_manager.complete_task(logged_in_user, int(task_id))
                        else:
                            print('Invalid task id')
                    elif choice1 == '5':
                        logged_in_user = ''
                        print('Logout successfully')
                        break
                    else:
                        print('Invalid choice')
            else:
                continue

        elif choice == '3':
            break
        else:
            print('Invalid choice')


if __name__ == '__main__':
    main()
