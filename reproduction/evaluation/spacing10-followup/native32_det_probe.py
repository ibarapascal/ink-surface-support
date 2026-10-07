"""Native32/legacy64 export-eligibility audit; frozen reference geometry stays float64."""
import hashlib,os
from pathlib import Path
import numpy as np
import torch
import dtypes
import model as native_model

MODEL_SHA = "21c6e85be5c394c70abeaa85a92d209b28cc443c40c3d7135b90463def0168a2"
FORMULA_SHA = "89c2b9a5b53530033208509c275cc6aa1314bb784589bc08f2c76ebbbb19532e"


def mask(uv_yx, source_xyz, source_cell_valid, numpy_float_hi):
    # Casts and expression block copied from locked native inversion source.
    uv = uv_yx.detach().cpu().numpy().astype(numpy_float_hi, copy=False)
    xyz = source_xyz.detach().cpu().numpy().astype(numpy_float_hi, copy=False)
    cell_valid = source_cell_valid.detach().cpu().numpy().astype(bool, copy=False)
    q00_all = uv[:-1, :-1]
    q10_all = uv[1:, :-1]
    q01_all = uv[:-1, 1:]
    q11_all = uv[1:, 1:]
    xyz00_all = xyz[:-1, :-1]
    xyz10_all = xyz[1:, :-1]
    xyz01_all = xyz[:-1, 1:]
    xyz11_all = xyz[1:, 1:]
    finite_uv = (
    	np.isfinite(q00_all).all(axis=-1) &
    	np.isfinite(q10_all).all(axis=-1) &
    	np.isfinite(q01_all).all(axis=-1) &
    	np.isfinite(q11_all).all(axis=-1)
    )
    finite_xyz = (
    	np.isfinite(xyz00_all).all(axis=-1) &
    	np.isfinite(xyz10_all).all(axis=-1) &
    	np.isfinite(xyz01_all).all(axis=-1) &
    	np.isfinite(xyz11_all).all(axis=-1)
    )
    tri0_s_all = q10_all - q00_all
    tri0_t_all = q01_all - q00_all
    tri1_u_all = q11_all - q10_all
    tri1_v_all = q01_all - q10_all
    det0 = tri0_s_all[..., 0] * tri0_t_all[..., 1] - tri0_s_all[..., 1] * tri0_t_all[..., 0]
    det1 = tri1_u_all[..., 0] * tri1_v_all[..., 1] - tri1_u_all[..., 1] * tri1_v_all[..., 0]
    valid_cells_2d = (
    	cell_valid
    	& finite_uv
    	& finite_xyz
    	& np.isfinite(det0)
    	& np.isfinite(det1)
    	& (det0 > 1.0e-10)
    	& (det1 > 1.0e-10)
    )
    return valid_cells_2d


def audit(checkpoint, old_report, diff_path):
    assert os.environ.get('LASAGNA_MAX_PRECISION_FLOAT') == '32'
    assert dtypes.torch_float_hi == torch.float32 and dtypes.numpy_float_hi == np.float32
    assert hashlib.sha256(Path(native_model.__file__).read_bytes()).hexdigest() == MODEL_SHA
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    parts = [state[k] for k in sorted((k for k in state if k.startswith('flatten_map_ms.')), key=lambda k:int(k.rsplit('.',1)[1]))]
    uv = native_model.Model3D._integrate_pyramid_3d(parts, pyramid_d=False)[:,0].permute(1,2,0).contiguous()
    xyz = state['flatten_source_xyz']; cells = state['flatten_source_cell_valid']
    actual = mask(uv, xyz, cells, np.float32); legacy = mask(uv, xyz, cells, np.float64)
    assert int(legacy.sum()) == old_report['source_export_eligible_quads']
    different = actual != legacy; count = int(different.sum())
    envkeys = ['LASAGNA_MAX_PRECISION_FLOAT','LASAGNA_COMPILE_FLATTEN','LASAGNA_FUSED_FLATTEN_ADAM_CLAMP','LASAGNA_FLATTEN_DIAGNOSTICS','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']
    record = {'native_model_sha256':MODEL_SHA,'native_formula_sha256':FORMULA_SHA,
      'dtype32_eligible_quads':int(actual.sum()),'legacy64_eligible_quads':int(legacy.sum()),
      'different_cell_count':count,'masks_equal':count==0,'shape':list(actual.shape),
      'dtype32_mask_sha256':hashlib.sha256(actual.tobytes()).hexdigest(),
      'legacy64_mask_sha256':hashlib.sha256(legacy.tobytes()).hexdigest(),
      'difference_positions_preview_rc':np.argwhere(different)[:32].tolist(),
      'effective_eval_runtime_env':{k:os.environ.get(k) for k in envkeys},
      'effective_module_dtype':{'torch_float_hi':str(dtypes.torch_float_hi),'numpy_float_hi':np.dtype(dtypes.numpy_float_hi).name,
        'torch_intraop_threads':torch.get_num_threads(),'torch_interop_threads':torch.get_num_interop_threads(),
        'source_xyz_dtype':str(xyz.dtype),'forward_uv_dtype':str(uv.dtype)},
      'diagnostic_dispatch':'Original eager core, no optimizer/compile/fused optimizer invoked',
      'reference_geometry_precision':'Unchanged float64 TriangleSoup/XYZ/areas/distances'}
    if count:
        # Complete recoverable locations, bounded bit masks, not a full-stage mesh.
        np.savez_compressed(diff_path, shape=np.asarray(actual.shape,dtype=np.int64),
            native32=np.packbits(actual.ravel()),legacy64=np.packbits(legacy.ravel()),
            different=np.packbits(different.ravel()))
        record['all_difference_positions']={'encoding':'np.unpackbits(different)[:prod(shape)].reshape(shape)',
           'path':str(diff_path),'sha256':hashlib.sha256(Path(diff_path).read_bytes()).hexdigest()}
        old_report['legacy64_eligibility_proxy_not_native']={
          'native_inversion_eligibility':old_report['native_inversion_eligibility'],
          'eligible_source_to_export':old_report['eligible_source_to_export'],
          'source_export_eligible_quads':old_report['source_export_eligible_quads'],
          'physical_area_mm2':old_report['physical_area_mm2']['source_export_eligible'],
          'stage_geometry':old_report['all_stage_geometry']['source_export_eligible']}
        pending={'status':'PENDING_NATIVE32_DIAGNOSTIC_QUALIFICATION','native32_eligible_quads':int(actual.sum()),
          'reason':'Legacy64 eligibility differs from actual native32; no quiet reinterpretation of source-to-export proxy.'}
        old_report['native_inversion_eligibility']=pending.copy();old_report['eligible_source_to_export']=pending.copy()
        old_report['source_export_eligible_quads']=None
        old_report['physical_area_mm2']['source_export_eligible']=None
        old_report['all_stage_geometry']['source_export_eligible']=pending.copy()
    old_report['native32_vs_legacy64_eligibility_audit']=record
    return record
