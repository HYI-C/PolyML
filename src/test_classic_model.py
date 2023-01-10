import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from run_autoencoder import run_model
from data_loader.sequence_datamodule import SequenceDataModule
#..from event_model.data.datasets.btyd import BTYD, GenMode
#..from event_model.models import TargetCreator
from model.auto_encoder import ClassicModel
from model.nets.embedder import ContinuousEmbedder

from ..test_datasets.datamodels import EventprofilesDataModel #..This is a set of labels of features

DISCRETE_START_TOKEN = "StartToken"
DEVICE_TYPE = "cpu"



def log_dir() -> str:
    with TemporaryDirectory() as f:
        yield f


def data_dir() -> str:
    return str(Path(__file__).parents[2] / "examples")


class TestClassicModel:
    def test_btyd(self, vae_sample_z: bool, log_dir: str, data_dir: str) -> None:
        # create btyd data and dependent modules
        data_model = EventprofilesDataModel()
        dataset = None
        """
        dataset = BTYD.from_modes(
            modes=[GenMode(a=1, b=3, r=1, alpha=15)],
            num_customers=100,
            seq_gen_dynamic=True,
            start_date=datetime.datetime(2019, 1, 1, 0, 0, 0),
            start_limit_date=datetime.datetime(2019, 6, 15, 0, 0, 0),
            end_date=datetime.datetime(2021, 1, 1, 0, 0, 0),
            data_dir=data_dir,
            continuous_features=data_model.cont_feat,
            discrete_features=data_model.discr_feat,
        ) #...We should regenerate this dataset and encode the vertex data here (should we do this with a list of lists?)
        """
        embedder = ContinuousEmbedder(
            continuous_features=dataset.continuous_feature_names,
            category_dict=dataset.get_discrete_feature_values(start_token=DISCRETE_START_TOKEN),
            embed_dim=256,
            drop_rate=0.5,
            pre_encoded=False,
        )

        target_transform = TargetCreator(
            cols=(data_model.target_cols + data_model.cont_feat),
            emb=embedder,
            max_item_len=100,
            start_token_discr=DISCRETE_START_TOKEN,
            start_token_cont=0,
        )

        datamodule = SequenceDataModule(
            dataset=dataset,
            target_transform=target_transform,
            test_size=0.2,
            batch_points=256,
            device=DEVICE_TYPE,
            min_points=1,
        )

        # create model
        model = ClassicModel(
            embedder,
            rnn_dim=256,
            drop_rate=0.5,
            bottleneck_dim=32,
            lr=0.001,
            target_cols=target_transform.cols,
            vae_sample_z=vae_sample_z,
            vae_sampling_scaler=1.0,
            vae_KL_weight=0.01,
        )

        run_model(
            datamodule,
            model,
            log_dir=log_dir,
            device_type=DEVICE_TYPE,
            num_epochs=2,
            val_check_interval=2,
            limit_val_batches=2,
        )