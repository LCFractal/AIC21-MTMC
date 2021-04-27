import numpy as np
import pandas as pd

def interpolate_traj(trks, mark_interpolation=False, drop_len=1):
    '''
    trks: 2d np array of MOT format.
    '''
    print('performing interpolation.')
    traj_df = pd.DataFrame(data=trks[:, :7], columns=['frame', 'trkid', 'x', 'y', 'w', 'h', 'score'])
    # Discard trajectories that appear few times
    short_tracklets_ix = traj_df.index[
        traj_df.groupby('trkid')['frame'].transform('count') <= drop_len]
    traj_df.drop(short_tracklets_ix, inplace=True)

    traj_df['cx'] = traj_df['x'] + 0.5 * traj_df['w']
    traj_df['cy'] = traj_df['y'] + 0.5 * traj_df['h']
    # Build sub_dfs with full trajectories across all missing frames
    reixed_traj_df = traj_df.set_index('trkid')
    full_traj_dfs = []
    traj_start_ends = traj_df.groupby('trkid')['frame'].agg(['min', 'max'])
    for ped_id, (traj_start, traj_end) in traj_start_ends.iterrows():
        if ped_id != -1:
            full_traj_df = pd.DataFrame(data=np.arange(traj_start, traj_end + 1), columns=['frame'])
            partial_traj_df = reixed_traj_df.loc[[ped_id]].reset_index()
            if mark_interpolation:
                partial_traj_df['flag'] = 1

                # Interpolate bb centers, heights and widths
                full_traj_df = pd.merge(full_traj_df,
                                    partial_traj_df[['trkid', 'frame', 'cx', 'cy', 'h', 'w', 'score', 'flag', ]],
                                    how='left', on='frame')
                full_traj_df['flag'].fillna(0, inplace=True)
            else:
                full_traj_df = pd.merge(full_traj_df,
                                    partial_traj_df[['trkid', 'frame', 'cx', 'cy', 'h', 'w', 'score']],
                                    how='left', on='frame')
            full_traj_df = full_traj_df.sort_values(by='frame').interpolate()
            remove_interp_prepare = False
            '''
            if mark_interpolation:
                index_to_drop = []
                for index, row in full_traj_df.iterrows():
                    if remove_interp_prepare and row['flag'] == 0:
                        index_to_drop.append(index)
                        remove_interp_prepare = True
                        continue
                    if row['score'] < 0.5 and row['flag'] == 1:
                        # remove next if interpolation.
                        remove_interp_prepare = True
                    else:
                        remove_interp_prepare = False
                for index, row in full_traj_df[::-1].iterrows():
                    if remove_interp_prepare and row['flag'] == 0:
                        index_to_drop.append(len(full_traj_df) - 1 - index)
                        remove_interp_prepare = True
                        continue
                    if row['score'] < 0.5 and row['flag'] == 1:
                        # remove next if interpolation.
                        remove_interp_prepare = True
                    else:
                        remove_interp_prepare = False
                full_traj_df.drop(index_to_drop, inplace=True)
            '''
            full_traj_dfs.append(full_traj_df)

    traj_df = pd.concat(full_traj_dfs)
    # Recompute bb coords based on the interpolated centers, heights and widths
    traj_df['x'] = traj_df['cx'] - 0.5 * traj_df['w']
    traj_df['y'] = traj_df['cy'] - 0.5 * traj_df['h']
    if mark_interpolation:
        return traj_df[['frame', 'trkid', 'x', 'y', 'w', 'h', 'score', 'flag']].to_numpy()
    else:
        return traj_df[['frame', 'trkid', 'x', 'y', 'w', 'h', 'score']].to_numpy()

def remove_len1_traj(trks, drop_len=1):
    '''
    trks: 2d np array of MOT format.
    '''
    print('removing len 1 tracks.')
    traj_df = pd.DataFrame(data=trks[:, :7], columns=['frame', 'trkid', 'x', 'y', 'w', 'h', 'score'])
    # Discard trajectories that appear few times
    short_tracklets_ix = traj_df.index[
        traj_df.groupby('trkid')['frame'].transform('count') <= drop_len]
    print('removing len 1 tracks: ', len(short_tracklets_ix))
    traj_df.drop(short_tracklets_ix, inplace=True)
    return traj_df[['frame', 'trkid', 'x', 'y', 'w', 'h', 'score']].to_numpy()
