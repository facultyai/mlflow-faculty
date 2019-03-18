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


from uuid import uuid4

import pytest

from mlflow_faculty.artifact_repository import (
    _S3ArtifactRepositoryWithClientOverride,
    FacultyDatasetsArtifactRepository,
)


PROJECT_ID = uuid4()
ARTIFACT_URI = "faculty-datasets:{}/path/in/datasets".format(PROJECT_ID)
S3_BUCKET_NAME = "faculty-datasets-s3-bucket"
S3_URI = "s3://{}/{}/path/in/datasets".format(S3_BUCKET_NAME, PROJECT_ID)


@pytest.fixture
def mock_session(mocker):
    session = mocker.Mock()
    session.bucket.return_value = S3_BUCKET_NAME
    mocker.patch("faculty.datasets.session.get", return_value=session)
    return session


@pytest.fixture
def mock_s3_repo(mocker):
    s3_repo = mocker.Mock()
    mocker.patch(
        "mlflow_faculty.artifact_repository."
        "_S3ArtifactRepositoryWithClientOverride",
        return_value=s3_repo,
    )
    return s3_repo


def test_s3_repo_with_client_override(mocker):

    artifact_uri = mocker.Mock()
    s3_client_factory = mocker.Mock()

    repo = _S3ArtifactRepositoryWithClientOverride(
        artifact_uri, s3_client_factory
    )

    assert repo.artifact_uri == artifact_uri
    assert repo._get_s3_client() == s3_client_factory.return_value


@pytest.mark.parametrize("suffix", ["", "/"])
def test_faculty_repo(mocker, mock_session, suffix):

    s3_repo = mocker.Mock()
    s3_repo_init = mocker.patch(
        "mlflow_faculty.artifact_repository."
        "_S3ArtifactRepositoryWithClientOverride",
        return_value=s3_repo,
    )

    repo = FacultyDatasetsArtifactRepository(ARTIFACT_URI + suffix)

    assert repo._s3_repository == s3_repo

    # Check arguments that the S3 repo was built with
    passed_uri, passed_client_factory = s3_repo_init.call_args[0]
    # The first argument should be the correct S3 URI
    assert passed_uri == S3_URI
    # The second argument should be a function that builds an S3 client with
    # the correct project ID
    assert passed_client_factory() == mock_session.s3_client.return_value
    mock_session.s3_client.assert_called_once_with(PROJECT_ID)


@pytest.mark.parametrize(
    "uri, message",
    [
        ("no/schema", "Not a Faculty datasets URI"),
        ("wrong-schema:", "Not a Faculty datasets URI"),
        (
            "faculty-datasets://{}/path/in/datasets".format(PROJECT_ID),
            "Invalid URI.*Did you mean '{}'".format(ARTIFACT_URI),
        ),
        ("faculty-datasets:invalid-uri", "is not a valid UUID"),
    ],
    ids=["No schema", "Wrong schema", "Double slash", "Invalid UUID"],
)
def test_faculty_repo_invalid_uri(uri, message):
    with pytest.raises(ValueError, match=message):
        FacultyDatasetsArtifactRepository(uri)


def test_faculty_repo_get_path_module(mock_session, mock_s3_repo):
    repo = FacultyDatasetsArtifactRepository(ARTIFACT_URI)
    assert repo.get_path_module() == mock_s3_repo.get_path_module.return_value
    mock_s3_repo.get_path_module.assert_called_once_with()


def test_faculty_repo_log_artifact(mocker, mock_session, mock_s3_repo):
    local_file = mocker.Mock()
    artifact_path = mocker.Mock()
    repo = FacultyDatasetsArtifactRepository(ARTIFACT_URI)

    returned = repo.log_artifact(local_file, artifact_path)

    assert returned == mock_s3_repo.log_artifact.return_value
    mock_s3_repo.log_artifact.assert_called_once_with(
        local_file, artifact_path
    )


def test_faculty_repo_log_artifacts(mocker, mock_session, mock_s3_repo):
    local_file = mocker.Mock()
    artifact_path = mocker.Mock()
    repo = FacultyDatasetsArtifactRepository(ARTIFACT_URI)

    returned = repo.log_artifacts(local_file, artifact_path)

    assert returned == mock_s3_repo.log_artifacts.return_value
    mock_s3_repo.log_artifacts.assert_called_once_with(
        local_file, artifact_path
    )


def test_faculty_repo_list_artifacts(mocker, mock_session, mock_s3_repo):
    path = mocker.Mock()
    repo = FacultyDatasetsArtifactRepository(ARTIFACT_URI)

    returned = repo.list_artifacts(path)

    assert returned == mock_s3_repo.list_artifacts.return_value
    mock_s3_repo.list_artifacts.assert_called_once_with(path)


def test_faculty_repo_download_file(mocker, mock_session, mock_s3_repo):
    remote_file_path = mocker.Mock()
    local_path = mocker.Mock()
    repo = FacultyDatasetsArtifactRepository(ARTIFACT_URI)

    returned = repo._download_file(remote_file_path, local_path)

    assert returned == mock_s3_repo._download_file.return_value
    mock_s3_repo._download_file.assert_called_once_with(
        remote_file_path, local_path
    )
