import datetime
import json
import os
import time


class TaskManager:
    """
    Task manager to add, view, delete and update any tasks for the user

    """

    def __init__(self):
        self.tasks = {}
        if os.path.isfile('tasks.json'):
            with open('tasks.json', 'r') as f:
                try:
                    self.tasks = json.load(f)
                except Exception as e:
                    print('Failed to load tasks.json')
                finally:
                    f.close()

    def add_task(self, username, task_description, status='Pending'):
        if username not in self.tasks:
            self.tasks[username] = []
        self.tasks[username].append({"id": int(time.time()), "description": task_description, "status": status,
                                     "created_on": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
        self.save_task()
        print('Task added successfully')

    def save_task(self):
        with open('tasks.json', 'w') as f:
            json.dump(self.tasks, f)

    def complete_task(self, username, task_id):
        if username in self.tasks:
            tasks = self.tasks[username]
            if len(tasks) != 0:
                for task in tasks:
                    if task['id'] == task_id:
                        task['status'] = 'Completed'
                        self.save_task()
                        print(f'Task: {task_id} marked as completed')
                        break
                else:
                    print('No such task found')
        else:
            print('No tasks found')

    def delete_task(self, username, task_id):
        if username in self.tasks:
            tasks = self.tasks[username]
            if len(tasks) != 0:
                for task in tasks:
                    if task['id'] == task_id:
                        tasks.remove(task)
                        self.save_task()
                        print('Task deleted successfully')
                        break
                else:
                    print('No such task found')
        else:
            print('No tasks found')

    def view_task(self, username):
        if username in self.tasks:
            return self.tasks[username]
        else:
            return None
