import ffmpeg


def get_decoder(video_file: str, use_gpu=True):
    """Return the decoder name for a given video."""
    probe = ffmpeg.probe(video_file)
    video_stream = next(s for s in probe['streams']
                        if s['codec_type'] == 'video')
    codec_name = video_stream['codec_name']
    # For AV1, force to use cpu decoding
    if codec_name == 'av1':
        return 'libdav1d'
    if use_gpu:
        return f"{codec_name}_cuvid"
    return codec_name


def get_avg_fps(video_path: str) -> float:
    """Get the average frame rate of the video."""
    probe = ffmpeg.probe(video_path)
    video_stream = next(s for s in probe['streams']
                        if s['codec_type'] == 'video')
    return eval(video_stream['avg_frame_rate'])
