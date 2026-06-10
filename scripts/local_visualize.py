import argparse
import glob
import importlib
import os
import os.path as osp
import subprocess

import mmcv
from mmcv import Config
from mmcv.utils import import_modules_from_strings
from mmdet3d.datasets import build_dataset
from nuscenes.nuscenes import NuScenes
from PIL import Image
from tools.analysis_tools import visual
from tqdm import tqdm


def parse_args():
	parser = argparse.ArgumentParser(description='Build BEV and camera combined visualizations')
	parser.add_argument('--config', required=True, help='config file path')
	parser.add_argument('--result', required=True, help='results pickle path')
	parser.add_argument('--vis-root', required=True, help='root directory for visualization outputs')
	parser.add_argument('--max-show-num', type=int, default=0, help='0 means all frames')
	parser.add_argument('--vis-score-threshold', type=float, default=0.35, help='score threshold for displayed predictions')
	parser.add_argument('--show-gt-cam', action='store_true', help='show the GT half of the camera visualization')
	return parser.parse_args()


def import_plugin_modules(cfg, config_path):
	if cfg.get('custom_imports', None):
		import_modules_from_strings(**cfg['custom_imports'])

	if hasattr(cfg, 'plugin') and cfg.plugin:
		if hasattr(cfg, 'plugin_dir'):
			plugin_dir = cfg.plugin_dir
			module_dir = osp.dirname(plugin_dir).split('/')
			module_path = module_dir[0]
			for part in module_dir[1:]:
				module_path = module_path + '.' + part
			importlib.import_module(module_path)
		else:
			module_dir = osp.dirname(config_path).split('/')
			module_path = module_dir[0]
			for part in module_dir[1:]:
				module_path = module_path + '.' + part
			importlib.import_module(module_path)


def main():
	args = parse_args()

	vis_dir = osp.join(args.vis_root, 'vis_results')
	combined_dir = osp.join(args.vis_root, 'vis_combined')
	video_dir = osp.join(args.vis_root, 'videos')
	video_name = 'bevformer.mp4'

	mmcv.mkdir_or_exist(vis_dir)
	mmcv.mkdir_or_exist(combined_dir)
	mmcv.mkdir_or_exist(video_dir)

	cfg = Config.fromfile(args.config)
	import_plugin_modules(cfg, args.config)

	cfg.data.test.test_mode = True
	dataset = build_dataset(cfg.data.test)
	results = mmcv.load(args.result)

	result_files, tmp_dir = dataset.format_results(
		results,
		jsonfile_prefix=osp.join(vis_dir, 'results_nusc'))
	if isinstance(result_files, dict):
		result_path = result_files.get('pts_bbox')
		if result_path is None:
			result_path = next(iter(result_files.values()))
	else:
		result_path = result_files

	pred_data = mmcv.load(result_path)
	nusc = NuScenes(version=dataset.version, dataroot=dataset.data_root, verbose=False)
	visual.nusc = nusc

	for sample_token, sample_results in pred_data['results'].items():
		pred_data['results'][sample_token] = [
			record for record in sample_results
			if float(record.get('detection_score', 0.0)) >= args.vis_score_threshold
		]

	sample_tokens = list(pred_data['results'].keys())
	if args.max_show_num > 0:
		sample_tokens = sample_tokens[:args.max_show_num]

	for index, sample_token in enumerate(tqdm(sample_tokens, desc='Rendering', ncols=80)):
		tqdm.write(f'Rendering sample token {sample_token}')
		frame_prefix = osp.join(vis_dir, f'{index:06d}_{sample_token}')
		print()
		visual.render_sample_data(sample_token, pred_data=pred_data, out_path=frame_prefix)

		bev_candidates = sorted(glob.glob(frame_prefix + '_bev*'))
		camera_candidates = sorted(glob.glob(frame_prefix + '_camera*'))
		if not bev_candidates:
			raise FileNotFoundError(f'No BEV render found for {frame_prefix}')
		if not camera_candidates:
			raise FileNotFoundError(f'No camera render found for {frame_prefix}')

		bev_path = bev_candidates[0]
		camera_path = camera_candidates[0]
		bev_img = Image.open(bev_path).convert('RGB')
		camera_img = Image.open(camera_path).convert('RGB')
		if not args.show_gt_cam:
			camera_img = camera_img.crop((0, 0, camera_img.width, camera_img.height // 2))
		bev_img = bev_img.resize((bev_img.width * 2, bev_img.height * 2), Image.BICUBIC)
		canvas_width = bev_img.width + camera_img.width
		canvas_height = max(bev_img.height, camera_img.height)
		canvas = Image.new('RGB', (canvas_width, canvas_height), (255, 255, 255))
		canvas.paste(camera_img, (0, 0))
		canvas.paste(bev_img, (camera_img.width, 0))
		canvas.save(osp.join(combined_dir, f'{index:06d}_{sample_token}_combined.png'))
		tqdm.write(f'Finished sample token {sample_token}')

	if tmp_dir is not None:
		tmp_dir.cleanup()

	frame_list = osp.join(video_dir, 'frame_list.txt')
	frame_paths = sorted(glob.glob(osp.join(combined_dir, '*_combined.png')))
	with open(frame_list, 'w', encoding='utf-8') as file_handle:
		for frame_path in frame_paths:
			file_handle.write(f"file '{osp.abspath(frame_path)}'\n")

	if not frame_paths:
		raise FileNotFoundError(f'No visualization frames were generated in {vis_dir}')

	ffmpeg_command = [
		'ffmpeg',
		'-y',
		'-f', 'concat',
		'-safe', '0',
		'-r', '10',
		'-i', frame_list,
		'-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
		'-c:v', 'libx264',
		'-pix_fmt', 'yuv420p',
		osp.join(video_dir, video_name),
	]
	subprocess.run(ffmpeg_command, check=True)


if __name__ == '__main__':
	main()
