from dataclasses import dataclass

import cv2
import numpy as np
import threestudio
import torch
from threestudio.models.background.base import BaseBackground
from threestudio.models.exporters.base import Exporter, ExporterOutput
from threestudio.models.geometry.base import BaseImplicitGeometry, BaseGeometry
from threestudio.models.materials.base import BaseMaterial
from threestudio.models.mesh import Mesh
from threestudio.utils.rasterize import NVDiffRasterizerContext
from threestudio.utils.base import BaseObject
from threestudio.utils.typing import *


@dataclass
class ExporterOutput:
    save_name: str
    save_type: str
    params: Dict[str, Any]


class Exporter(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        save_video: bool = False

    cfg: Config

    def configure(
        self,
        geometry: BaseImplicitGeometry,
        material: BaseMaterial,
        background: BaseBackground,
    ) -> None:
        @dataclass
        class SubModules:
            geometry: BaseImplicitGeometry
            material: BaseMaterial
            background: BaseBackground

        self.sub_modules = SubModules(geometry, material, background)

    @property
    def geometry(self) -> BaseImplicitGeometry:
        return self.sub_modules.geometry

    @property
    def material(self) -> BaseMaterial:
        return self.sub_modules.material

    @property
    def background(self) -> BaseBackground:
        return self.sub_modules.background

    def __call__(self, *args, **kwargs) -> List[ExporterOutput]:
        raise NotImplementedError


@threestudio.register("dummy-exporter")
class DummyExporter(Exporter):
    def __call__(self, *args, **kwargs) -> List[ExporterOutput]:
        # DummyExporter does not export anything
        return []


@threestudio.register("gaussian-mesh-exporter")
class MeshExporter(Exporter):
    @dataclass
    class Config(Exporter.Config):
        fmt: str = "obj"
        save_name: str = "model"
        save_video: bool = True

    cfg: Config

    def configure(
        self,
        geometry: BaseGeometry,
        material: BaseMaterial,
        background: BaseBackground,
    ) -> None:
        super().configure(geometry, material, background)

    def __call__(self) -> List[ExporterOutput]:
        mesh: Mesh = self.geometry.extract_mesh()
        return self.export_obj(mesh)

    def export_obj(self, mesh: Mesh) -> List[ExporterOutput]:
        params = {"mesh": mesh}
        return [
            ExporterOutput(
                save_name=f"{self.cfg.save_name}.obj", save_type="obj", params=params
            )
        ]
