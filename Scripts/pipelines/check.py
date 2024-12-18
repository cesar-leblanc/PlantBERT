from utils import check_files, check_dependencies, check_environment

REQUIRED_DEPENDENCIES = {
    'accelerate':'0.25.0',
    'datasets':'2.16.1',
    'evaluate':'0.4.1',
    'numpy':'1.26.3',
    'openpyxl':'3.1.2',
    'pandas':'2.1.4',
    'requests':'2.31.0',
    'scikit-learn':'1.3.2',
    'setuptools':'57.4.0',
    'torch':'2.1.2',
    'tqdm':'4.66.1',
    'transformers':'4.36.2',
    'verde':'1.8.0'
}

class InstallationCheck:
    def files_check(self, accelerator):
        missing_files = check_files(accelerator)
        if missing_files:
            accelerator.print("The following files are missing:")
            for file in missing_files:
                accelerator.print(f"- {file}")
        else:
            accelerator.print("Files are all present.")

    def dependencies_check(self, accelerator, REQUIRED_DEPENDENCIES):
        missing_dependencies = check_dependencies(REQUIRED_DEPENDENCIES)
        if missing_dependencies:
            accelerator.print("The following dependencies are not correctly installed:")
            for dependency, required_version in missing_dependencies.items():
                accelerator.print(f"- {dependency} (required version: {required_version})")
        else:
            accelerator.print("Dependencies are correctly installed.")

    def environment_check(self, accelerator):
        missing_environments = check_environment()
        environment_errors = []
        if missing_environments[0]:
            environment_errors.append("The Python version is incompatible with the required version")
        if missing_environments[1]:
            environment_errors.append("CUDA is not installed")
        if missing_environments[2]:
            environment_errors.append("pip is not installed")
        if missing_environments[3]:
            environment_errors.append("Git is not installed")
        if missing_environments[4]:
            environment_errors.append("Git LFS is not installed")
        if missing_environments[5]:
            environment_errors.append("The pl@ntbert environment is not activated")
        if any(missing_environments):
            accelerator.print(f"Environment is not properly configured ({', '.join(environment_errors)}).")
        else:
            accelerator.print("Environment is properly configured.")

    def run(self, args, accelerator):
        accelerator.print('\n' + '*'*9)
        accelerator.print(f'* Check *')
        accelerator.print('*'*9 + '\n')
        if args.check_files:
            self.files_check(accelerator)
        else:
            accelerator.print("Not checking if files are missing.")
        if args.check_dependencies:
            self.dependencies_check(accelerator, REQUIRED_DEPENDENCIES)
        else:
            accelerator.print("Not checking if dependencies are missing.")
        if args.check_environment:
            self.environment_check(accelerator)
        else:
            accelerator.print("Not checking if the environment is correctly set up.")