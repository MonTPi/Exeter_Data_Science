import subprocess

def git_go():
    subprocess.call(['git', 'add', '.'])
    subprocess.call(['git','status'])
    subprocess.call(['git', 'commit', '-m' , f'{str(input())}'])
    subprocess.call(['git','status'])
    subprocess.call(['git', 'push', 'origin', 'main'])
    subprocess.call(['git','status'])
    


if __name__ == "__main__":
    git_go()
