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


from uuid import UUID
from functools import partial

from six.moves import urllib

from faculty.datasets import session
from mlflow.store.artifact_repo import ArtifactRepository
from mlflow.store.s3_artifact_repo import S3ArtifactRepository


class _S3ArtifactRepositoryWithClientOverride(S3ArtifactRepository):
    def __init__(self, artifact_uri, client_factory):
        super(_S3ArtifactRepositoryWithClientOverride, self).__init__(
            artifact_uri
        )
        self._client_factory = client_factory

    def _get_s3_client(self):
        return self._client_factory()


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

        cleaned_path = parsed_uri.path.strip("/")
        first_part = cleaned_path.split("/")[0]
        try:
            project_id = UUID(first_part)
        except ValueError:
            raise ValueError(
                "{} in given URI {} is not a valid UUID".format(
                    first_part, artifact_uri
                )
            )

        self._session = session.get()
        s3_bucket = self._session.bucket(project_id)
        s3_uri = "s3://{}/{}".format(s3_bucket, cleaned_path)
        s3_client_factory = partial(self._session.s3_client, project_id)

        self._s3_repository = _S3ArtifactRepositoryWithClientOverride(
            s3_uri, s3_client_factory
        )

    def get_path_module(self):
        return self._s3_repository.get_path_module()

    def log_artifact(self, local_file, artifact_path=None):
        return self._s3_repository.log_artifact(local_file, artifact_path)

    def log_artifacts(self, local_dir, artifact_path=None):
        return self._s3_repository.log_artifacts(local_dir, artifact_path)

    def list_artifacts(self, path):
        return self._s3_repository.list_artifacts(path)

    def _download_file(self, remote_file_path, local_path):
        return self._s3_repository._download_file(remote_file_path, local_path)
