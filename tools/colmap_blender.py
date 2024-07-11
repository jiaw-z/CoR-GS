import os
import numpy as np
import sys
import sqlite3
import json
from pathlib import Path
from tqdm import tqdm
import math


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

IS_PYTHON3 = sys.version_info[0] >= 3
MAX_IMAGE_ID = 2**31 - 1

CREATE_CAMERAS_TABLE = """CREATE TABLE IF NOT EXISTS cameras (
    camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    model INTEGER NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    params BLOB,
    prior_focal_length INTEGER NOT NULL)"""

CREATE_DESCRIPTORS_TABLE = """CREATE TABLE IF NOT EXISTS descriptors (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)"""

CREATE_IMAGES_TABLE = """CREATE TABLE IF NOT EXISTS images (
    image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    name TEXT NOT NULL UNIQUE,
    camera_id INTEGER NOT NULL,
    prior_qw REAL,
    prior_qx REAL,
    prior_qy REAL,
    prior_qz REAL,
    prior_tx REAL,
    prior_ty REAL,
    prior_tz REAL,
    CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < {}),
    FOREIGN KEY(camera_id) REFERENCES cameras(camera_id))
""".format(MAX_IMAGE_ID)

CREATE_TWO_VIEW_GEOMETRIES_TABLE = """
CREATE TABLE IF NOT EXISTS two_view_geometries (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    config INTEGER NOT NULL,
    F BLOB,
    E BLOB,
    H BLOB,
    qvec BLOB,
    tvec BLOB)
"""

CREATE_KEYPOINTS_TABLE = """CREATE TABLE IF NOT EXISTS keypoints (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)
"""

CREATE_MATCHES_TABLE = """CREATE TABLE IF NOT EXISTS matches (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB)"""

CREATE_NAME_INDEX = \
    "CREATE UNIQUE INDEX IF NOT EXISTS index_name ON images(name)"

CREATE_ALL = "; ".join([
    CREATE_CAMERAS_TABLE,
    CREATE_IMAGES_TABLE,
    CREATE_KEYPOINTS_TABLE,
    CREATE_DESCRIPTORS_TABLE,
    CREATE_MATCHES_TABLE,
    CREATE_TWO_VIEW_GEOMETRIES_TABLE,
    CREATE_NAME_INDEX
])


def array_to_blob(array):
    if IS_PYTHON3:
        return array.tostring()
    else:
        return np.getbuffer(array)

def blob_to_array(blob, dtype, shape=(-1,)):
    if IS_PYTHON3:
        return np.fromstring(blob, dtype=dtype).reshape(*shape)
    else:
        return np.frombuffer(blob, dtype=dtype).reshape(*shape)

class COLMAPDatabase(sqlite3.Connection):

    @staticmethod
    def connect(database_path):
        return sqlite3.connect(database_path, factory=COLMAPDatabase)

    def __init__(self, *args, **kwargs):
        super(COLMAPDatabase, self).__init__(*args, **kwargs)

        self.create_tables = lambda: self.executescript(CREATE_ALL)
        self.create_cameras_table = \
            lambda: self.executescript(CREATE_CAMERAS_TABLE)
        self.create_descriptors_table = \
            lambda: self.executescript(CREATE_DESCRIPTORS_TABLE)
        self.create_images_table = \
            lambda: self.executescript(CREATE_IMAGES_TABLE)
        self.create_two_view_geometries_table = \
            lambda: self.executescript(CREATE_TWO_VIEW_GEOMETRIES_TABLE)
        self.create_keypoints_table = \
            lambda: self.executescript(CREATE_KEYPOINTS_TABLE)
        self.create_matches_table = \
            lambda: self.executescript(CREATE_MATCHES_TABLE)
        self.create_name_index = lambda: self.executescript(CREATE_NAME_INDEX)

    def update_camera(self, model, width, height, params, camera_id):
        params = np.asarray(params, np.float64)
        cursor = self.execute(
            "UPDATE cameras SET model=?, width=?, height=?, params=?, prior_focal_length=1 WHERE camera_id=?",
            (model, width, height, array_to_blob(params),camera_id))
        return cursor.lastrowid

def round_python3(number):
    rounded = round(number)
    if abs(number - rounded) == 0.5:
        return 2.0 * round(number / 2.0)
    return rounded

def pipeline(scene, base_path, n_views):
    llffhold = 8
    view_path = str(n_views) + '_views'
    os.chdir(base_path + scene)
    os.system('rm -r ' + view_path)
    os.mkdir(view_path)
    os.chdir(view_path)
    os.mkdir('created')
    os.mkdir('triangulated')
    os.mkdir('images')
    os.system('colmap model_converter  --input_path ../sparse/0/ --output_path ../sparse/0/  --output_type TXT')


    images = {}

    with open('../transforms_train.json') as json_file:
        contents = json.load(json_file)
        # print(f"contents is {contents}")
        fovx = contents["camera_angle_x"]
        # print(f"fovx is {fovx}")
        skip = 1
        frames = contents["frames"][::skip]
        idx = 1
        for frame in frames:
            fname = frame['file_path'].split('/')[-1]
            if not (fname.endswith('.png') or fname.endswith('.jpg')):
                fname += '.png'
            pose = np.array(frame['transform_matrix']) @ blender2opencv
            R = np.linalg.inv(pose[:3, :3])
            T = -np.matmul(R, pose[:3, 3])
            q0 = 0.5 * math.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2])
            q1 = (R[2, 1] - R[1, 2]) / (4 * q0)
            q2 = (R[0, 2] - R[2, 0]) / (4 * q0)
            q3 = (R[1, 0] - R[0, 1]) / (4 * q0)
            elems = f'{idx} {q0} {q1} {q2} {q3} {T[0]} {T[1]} {T[2]} 1 {fname}\n\n'
            images[fname] = elems[1:]

    train_idx = [2, 16, 26, 55, 73, 76, 86, 93]
        
    # img_list = sorted(images.keys(), key=lambda x: x)
    img_list = images.keys()
    # train_img_list = [c for idx, c in enumerate(img_list) if idx % llffhold != 0]
    train_img_list = [c for idx, c in enumerate(img_list)]
    if n_views > 0:
        # idx_sub = [round_python3(i) for i in np.linspace(0, len(train_img_list)-1, n_views)]
        idx_sub = train_idx[:n_views]
        train_img_list = [c for idx, c in enumerate(train_img_list) if idx in idx_sub]

    print(f"train_img_list is {train_img_list}")

    for img_name in train_img_list:
        os.system('cp ../images/' + img_name + '  images/' + img_name)

    os.system('cp ../blender2colmap/sparse/cameras.txt created/.')
    with open('created/points3D.txt', "w") as fid:
        pass

    res = os.popen( 'colmap feature_extractor --database_path database.db --image_path images  --SiftExtraction.max_image_size 4032 --SiftExtraction.max_num_features 16384 --SiftExtraction.estimate_affine_shape 1 --SiftExtraction.domain_size_pooling 1').read()
    print(f"feature load")
    os.system( 'colmap exhaustive_matcher --database_path database.db --SiftMatching.guided_matching 1 --SiftMatching.max_num_matches 32768')
    db = COLMAPDatabase.connect('database.db')
    db_images = db.execute("SELECT * FROM images")
    img_rank = [db_image[1] for db_image in db_images]
    print(img_rank, res)
    with open('created/images.txt', "w") as fid:
        for idx, img_name in enumerate(img_rank):
            print(img_name)
            # data = [str(1 + idx)] + [' ' + item for item in images[os.path.basename(img_name)]] + ['\n\n']
            data = [str(1 + idx)] + [images[os.path.basename(img_name)]]
            fid.writelines(data)

    os.system('colmap point_triangulator --database_path database.db --image_path images --input_path created  --output_path triangulated  --Mapper.ba_local_max_num_iterations 40 --Mapper.ba_local_max_refinements 3 --Mapper.ba_global_max_num_iterations 100')
    os.system('colmap model_converter  --input_path triangulated --output_path triangulated  --output_type TXT')
    os.system('colmap image_undistorter --image_path images --input_path triangulated --output_path dense')
    os.system('colmap patch_match_stereo --workspace_path dense')
    os.system('colmap stereo_fusion --workspace_path dense --output_path dense/fused.ply')


# for scene in ['bicycle', 'bonsai', 'counter', 'garden',  'kitchen', 'room', 'stump']:
# for scene in ['bicycle', 'bonsai', 'counter', 'garden', 'kitchen', 'room', 'stump']:
# ['chair', 'drums', 'ficus', 'hotdog', 'lego', 'materials', 'mic', 'ship']
scene = 'drums'
pipeline(scene, base_path = 'data/nerf_synthetic/', n_views = 8)


