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


import os
import posixpath
from uuid import UUID

from six.moves import urllib
import faculty
from faculty import datasets
from mlflow.store.artifact_repo import ArtifactRepository

from mlflow_faculty.converters import faculty_object_to_mlflow_file_info


class FacultyDatasetsArtifactRepository(ArtifactRepository):
    def __init__(self, artifact_uri):
        parsed_uri = urllib.parse.urlparse(artifact_uri)
        if parsed_uri.scheme != "faculty-datasets":
            raise ValueError(
                "Not a Faculty datasets URI: {}".format(artifact_uri)
            )
        # Test for PROJECT_ID in netloc rather than path.
        elif parsed_uri.netloc != "":
            raise ValueError(
                "Invalid URI {}. Netloc is reserved. "
                "Did you mean 'faculty-datasets:{}{}'".format(
                    artifact_uri, parsed_uri.netloc, parsed_uri.path
                )
            )

        cleaned_path = parsed_uri.path.strip("/") + "/"
        first_part, remainder = cleaned_path.split("/", 1)

        try:
            self.project_id = UUID(first_part)
        except ValueError:
            raise ValueError(
                "{} in given URI {} is not a valid UUID".format(
                    first_part, artifact_uri
                )
            )

        self.datasets_artifact_root = "/" + remainder

    def _datasets_path(self, artifact_path):
        return posixpath.normpath(
            posixpath.join(
                self.datasets_artifact_root, artifact_path.lstrip("/")
            )
        )

    def log_artifact(self, local_file, artifact_path=None):
        if artifact_path is None:
            artifact_path = os.path.basename(local_file)
        datasets_path = self._datasets_path(artifact_path)
        datasets.put(local_file, datasets_path, self.project_id)

    def log_artifacts(self, local_dir, artifact_path=None):
        if artifact_path is None:
            artifact_path = "./"
        datasets_path = self._datasets_path(artifact_path)
        datasets.put(local_dir, datasets_path, self.project_id)

    def list_artifacts(self, path=None):
        if path is None:
            path = "./"
        datasets_path = self._datasets_path(path)

        # Make sure path interpreted as a directory
        prefix = datasets_path.rstrip("/") + "/"

        # Go directly to the object store so we can get file sizes in the
        # response
        client = faculty.client("object")

        list_response = client.list(self.project_id, prefix)
        objects = list_response.objects

        while list_response.next_page_token is not None:
            list_response = client.list(
                self.project_id, prefix, list_response.next_page_token
            )
            objects += list_response.objects

        infos = [
            faculty_object_to_mlflow_file_info(
                obj, self.datasets_artifact_root
            )
            for obj in objects
        ]

        # Remove root
        return [i for i in infos if i.path != "/"]

    def _download_file(self, remote_file_path, local_path):
        datasets_path = self._datasets_path(remote_file_path)
        datasets.get(datasets_path, local_path, self.project_id)
