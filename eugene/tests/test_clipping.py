import eugene as eu
import numpy as np

def test_clip_segments_1d():
    # generate some data
    t = np.linspace(0.,10.,1000)
    x = np.exp(0.1 * t)
    y = np.exp(0.2 * t)
    A = [x] * 10
    B = [y] * 10

    # clip
    A_clipped, B_clipped = eu.clipping.clip_segments(A, B, 10)

    assert len(B_clipped[0]) < len(A_clipped[0])
    for seg in A:
        assert seg.shape == A[0].shape
    for seg in B:
        assert seg.shape == B[0].shape


def test_clip_segments():
    # generate some data
    t = np.linspace(0.,10.,1000).reshape(1, -1) * np.ones((5, 1000))
    x = np.exp(0.1 * t)
    y = np.exp(0.2 * t)
    A = [x] * 10
    B = [y] * 10

    # clip
    A_clipped, B_clipped = eu.clipping.clip_segments(A, B, 10)

    assert B_clipped[0].shape[1] < A_clipped[0].shape[1]
    for seg in A:
        assert seg.shape == A[0].shape
    for seg in B:
        assert seg.shape == B[0].shape


def test_clip_to_match():
    A = [np.ones((3,100))] * 5
    B = [np.zeros((3,50))] * 5
    C = [np.ones((3,100))] * 5
    D = [np.ones((3,100))] * 5

    C_clipped, D_clipped = eu.clipping.clip_to_match(A, B, C, D)

    for i, seg in enumerate(A):
        assert seg.shape == C_clipped[i].shape

    for i, seg in enumerate(B):
        assert seg.shape == D_clipped[i].shape

    assert D_clipped[0].shape == (3,50)


def test_clip_to_match_1D():
    A = [np.ones(100)] * 5
    B = [np.zeros(50)] * 5
    C = [np.ones(100)] * 5
    D = [np.ones(100)] * 5

    C_clipped, D_clipped = eu.clipping.clip_to_match(A, B, C, D)

    for i, seg in enumerate(A):
        assert len(seg) == len(C_clipped[i])

    for i, seg in enumerate(B):
        assert len(seg) == len(D_clipped[i])

    assert len(D_clipped[0]) == 50


def test_zip_curves():
    A = [np.ones((3,6)), np.ones((3,6))]
    B = [np.zeros((3,6)), np.zeros((3,6))]
    C = [np.zeros((3,6)), np.zeros((3,6))]
    D = [np.ones((3,6)), np.ones((3,6))]

    data1, data2 = eu.clipping.zip_curves(A, B, C, D)

    expected1 = np.concatenate([np.ones((3,12)), np.zeros((3,12))], axis=0)
    expected2 = np.concatenate([np.zeros((3,12)), np.ones((3,12))], axis=0)

    assert np.all(data1 == expected1)
    assert np.all(data2 == expected2)


def test_zip_curves_1d():
    A = [np.ones(6), np.ones(6)]
    B = [np.zeros(6), np.zeros(6)]
    C = [np.zeros(6), np.zeros(6)]
    D = [np.ones(6), np.ones(6)]

    data1, data2 = eu.clipping.zip_curves(A, B, C, D)

    expected1 = np.concatenate([np.ones((1,12)), np.zeros((1,12))], axis=0)
    expected2 = np.concatenate([np.zeros((1,12)), np.ones((1,12))], axis=0)

    assert np.all(data1 == expected1)
    assert np.all(data2 == expected2)
