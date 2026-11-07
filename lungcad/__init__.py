"""lungcad — code for the paired evaluation of dose and blur augmentation for a 3D RetinaNet lung-nodule detector.

Package layout:
  lungcad.config        paths + the constants every stage shares (HU window, spacings)
  lungcad.preprocess    the one shared preprocessing module (LUNA16 + NLST)
  lungcad.augment       dose / blur (kernel) augmentation of the augmented arm (S4)
  lungcad.data          LUNA16 index and preprocessing driver, NLST inventory / conversion / loader
  lungcad.stage1        RetinaNet 3D detector: model, dataset, train, infer (S3, S4)
  lungcad.eval          official-LUNA16-equivalent FROC/CPM scorer + paired bootstrap
"""
__version__ = "0.1.0"
