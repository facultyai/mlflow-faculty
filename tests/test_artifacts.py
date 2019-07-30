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

import faculty
import faculty.datasets
from mlflow_faculty.artifacts import FacultyDatasetsArtifactRepository


PROJECT_ID = uuid4()
ARTIFACT_URI = "faculty-datasets:{}/path/in/datasets".format(PROJECT_ID)
ARTIFACT_ROOT = "/path/in/datasets/"


@pytest.mark.parametrize("suffix", ["", "/"])
def test_faculty_repo_init(suffix):
    repo = FacultyDatasetsArtifactRepository(ARTIFACT_URI + suffix)
    assert repo.project_id == PROJECT_ID
    assert repo.datasets_artifact_root == ARTIFACT_ROOT


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


@pytest.mark.parametrize("prefix", ["", "/"])
def test_faculty_repo_log_artifact(mocker, prefix):
    mocker.patch("faculty.datasets.put")

    repo = FacultyDatasetsArtifactRepository(ARTIFACT_URI)
    repo.log_artifact("/local/file.txt", prefix + "remote/object.txt")

    faculty.datasets.put.assert_called_once_with(
        "/local/file.txt", ARTIFACT_ROOT + "remote/object.txt", PROJECT_ID
    )


def test_faculty_repo_log_artifact_default_destination(mocker):
    mocker.patch("faculty.datasets.put")

    repo = FacultyDatasetsArtifactRepository(ARTIFACT_URI)
    repo.log_artifact("/local/file.txt")

    faculty.datasets.put.assert_called_once_with(
        "/local/file.txt", ARTIFACT_ROOT + "file.txt", PROJECT_ID
    )


@pytest.mark.parametrize("prefix", ["", "/"])
def test_faculty_repo_log_artifacts(mocker, prefix):
    mocker.patch("faculty.datasets.put")

    repo = FacultyDatasetsArtifactRepository(ARTIFACT_URI)
    repo.log_artifacts("/local/dir", prefix + "remote/folder")

    faculty.datasets.put.assert_called_once_with(
        "/local/dir", ARTIFACT_ROOT + "remote/folder", PROJECT_ID
    )


def test_faculty_repo_log_artifacts_default_destination(mocker):
    mocker.patch("faculty.datasets.put")

    repo = FacultyDatasetsArtifactRepository(ARTIFACT_URI)
    repo.log_artifacts("/local/dir")

    faculty.datasets.put.assert_called_once_with(
        "/local/dir", ARTIFACT_ROOT.rstrip("/"), PROJECT_ID
    )


@pytest.mark.parametrize("prefix", ["", "/"])
@pytest.mark.parametrize("suffix", ["", "/"])
def test_faculty_repo_list_artifacts(mocker, prefix, suffix):
    objects = [mocker.Mock() for _ in range(10)]
    list_response_0 = mocker.Mock(objects=objects[:5], next_page_token="token")
    list_response_1 = mocker.Mock(objects=objects[5:], next_page_token=None)

    client = mocker.Mock()
    client.list.side_effect = [list_response_0, list_response_1]

    mocker.patch("faculty.client", return_value=client)

    mock_file_infos = [mocker.Mock() for _ in objects]
    mock_file_infos[-1].path = "/"
    converter_mock = mocker.patch(
        "mlflow_faculty.artifacts.faculty_object_to_mlflow_file_info",
        side_effect=mock_file_infos,
    )

    repo = FacultyDatasetsArtifactRepository(ARTIFACT_URI)
    assert (
        repo.list_artifacts(prefix + "a/dir" + suffix) == mock_file_infos[:-1]
    )

    faculty.client.assert_called_once_with("object")
    client.list.assert_has_calls(
        [
            mocker.call(PROJECT_ID, ARTIFACT_ROOT + "a/dir/"),
            mocker.call(PROJECT_ID, ARTIFACT_ROOT + "a/dir/", "token"),
        ]
    )
    converter_mock.assert_has_calls(
        [mocker.call(obj, ARTIFACT_ROOT) for obj in objects]
    )


def test_faculty_repo_list_artifacts_default_path(mocker):
    list_response = mocker.Mock(objects=[], next_page_token=None)

    client = mocker.Mock()
    client.list.side_effect = [list_response]

    mocker.patch("faculty.client", return_value=client)

    repo = FacultyDatasetsArtifactRepository(ARTIFACT_URI)
    assert repo.list_artifacts() == []

    faculty.client.assert_called_once_with("object")
    client.list.assert_called_once_with(PROJECT_ID, ARTIFACT_ROOT)


@pytest.mark.parametrize("prefix", ["", "/"])
def test_faculty_repo_download_file(mocker, prefix):
    mocker.patch("faculty.datasets.get")

    repo = FacultyDatasetsArtifactRepository(ARTIFACT_URI)
    repo._download_file(prefix + "path/to/file", "/local/path")

    faculty.datasets.get.assert_called_once_with(
        ARTIFACT_ROOT + "path/to/file", "/local/path", PROJECT_ID
    )
