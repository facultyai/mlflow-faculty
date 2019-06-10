# Copyright 2019 Faculty Science Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from setuptools import setup, find_packages


TRACKING_STORE_ENTRYPOINT = "faculty=mlflow_faculty:FacultyRestStore"
ARTIFACT_REPOSITORY_ENTRYPOINT = (
    "faculty-datasets=mlflow_faculty:FacultyDatasetsArtifactRepository"
)
RUN_CONTEXT_ENTRYPOINT = (
    "faculty-run-context=mlflow_faculty.context:FacultyRunContext"
)


setup(
    name="mlflow-faculty",
    description="MLflow plugin for the Faculty platform.",
    url="https://faculty.ai/",
    author="Faculty",
    author_email="opensource@faculty.ai",
    license="Apache Software License",
    packages=find_packages(),
    use_scm_version={"version_scheme": "post-release"},
    setup_requires=["setuptools_scm"],
    install_requires=["mlflow>=1.0.0", "faculty>=0.23.2", "six", "pytz"],
    entry_points={
        "mlflow.tracking_store": TRACKING_STORE_ENTRYPOINT,
        "mlflow.artifact_repository": ARTIFACT_REPOSITORY_ENTRYPOINT,
        "mlflow.run_context_provider": RUN_CONTEXT_ENTRYPOINT,
    },
)
