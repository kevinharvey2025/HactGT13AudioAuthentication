"""(uid, view) -> the exact float32 audio a model sees, plus its channel parameters.

View 0 is the clean canonical clip; views >= 1 add a random augmentation chain. All randomness
comes from uid_rng(uid, view), so every module (SSL, resynthesis, diffusion) gets identical audio
for the same (uid, view). Test clips are never cropped.
"""
from . import audio, augment


class ViewMaker:
    def __init__(self, babble_pool=None):
        self.durations = audio.TestLikeDurations.from_cache()
        self.babble_pool = babble_pool

    def __call__(self, uid, view=0, is_test=False):
        params = {"channel": "clean"}
        aug = None
        if view > 0:
            def aug(y, r):
                y2, p = augment.random_chain(y, r, babble_pool=self.babble_pool)
                params.update(p)
                return y2, p
        x = audio.canonical(audio.load_cached(uid), audio.uid_rng(uid, view),
                            None if is_test else self.durations, aug=aug)
        return x, params


def babble_pool(man, n=300, seed=0):
    """Real speech only (LibriSpeech speakers + LJSpeech), so background talkers never carry spoof audio."""
    real = man[(man.label == 0)].sample(n, random_state=seed)
    return [audio.load_cached(u) for u in real.uid]


def rng_for(uid, view):
    return audio.uid_rng(uid, view, seed=1)  # separate stream for module-level randomness (e.g. diffusion noise)

