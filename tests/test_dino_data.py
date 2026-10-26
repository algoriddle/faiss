# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""CPU checks for the original mmap-backed DINO loader; no Faiss/GPU needed."""

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np


_path = Path(__file__).resolve().parents[1] / "demos/e_means_10B/dino_data.py"
_spec = importlib.util.spec_from_file_location("dino_data", _path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
DinoBvecs = _module.DinoBvecs


class TestDinoBvecs(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.chunks = self.root / "chunked_base_10B"
        self.chunks.mkdir()
        self.data = np.arange(65, dtype=np.uint8).reshape(13, 5)
        self.write("00.bvecs", self.data[:4])
        self.write("01.bvecs", self.data[4:])

    def write(self, name, rows):
        with (self.chunks / name).open("wb") as stream:
            for row in rows:
                stream.write(len(row).to_bytes(4, "little", signed=True))
                stream.write(row.tobytes())

    def test_boundary_prefix_and_owned_output(self):
        source = DinoBvecs(self.root, n=11)
        self.assertEqual(source.n, 11)
        self.assertEqual(source.ends[-1], 13)
        self.assertEqual(len(source.maps), 2)
        self.assertTrue(all(isinstance(mapping, np.memmap) for mapping in source.maps))
        self.assertTrue(all(not mapping.flags.writeable for mapping in source.maps))
        np.testing.assert_array_equal(source.read(2, 7), self.data[2:7])
        output = source.read(2, 7)
        self.assertTrue(output.flags.owndata)
        output.fill(0)
        np.testing.assert_array_equal(source.read(2, 7), self.data[2:7])
        self.assertEqual(source.read(11, 11).shape, (0, 5))
        with self.assertRaises(IndexError):
            source.read(10, 12)

    def test_contiguous_reads_cover_prefix_without_aliasing(self):
        source = DinoBvecs(self.root, n=13)
        parts = [source.read(a, min(a + 3, source.n)) for a in range(0, source.n, 3)]
        self.assertEqual([len(part) for part in parts], [3, 3, 3, 3, 1])
        self.assertTrue(all(part.flags.owndata for part in parts))
        np.testing.assert_array_equal(np.concatenate(parts), self.data)
        parts[1].fill(0)
        np.testing.assert_array_equal(source.read(3, 6), self.data[3:6])

    def test_malformed_file_and_first_dimension(self):
        (self.chunks / "01.bvecs").write_bytes(b"\x05\x00")
        with self.assertRaisesRegex(ValueError, "incomplete"):
            DinoBvecs(self.root, n=13)
        (self.chunks / "01.bvecs").write_bytes(b"\x05\x00\x00\x00\x01")
        with self.assertRaisesRegex(ValueError, "incomplete"):
            DinoBvecs(self.root, n=13)
        (self.chunks / "01.bvecs").write_bytes(b"\x05\x00\x00\x00\x01\x02\x03\x04\x05")
        # A whole row is still valid; the requested population exceeds capacity.
        with self.assertRaisesRegex(ValueError, "corpus contains"):
            DinoBvecs(self.root, n=13)
        with (self.chunks / "00.bvecs").open("r+b") as stream:
            stream.write((0).to_bytes(4, "little", signed=True))
        with self.assertRaisesRegex(ValueError, "invalid bvecs dimension"):
            DinoBvecs(self.root, n=4)

    def test_population_and_range_validation(self):
        for n in (0, -1, 14):
            with self.assertRaises(ValueError):
                DinoBvecs(self.root, n=n)
        source = DinoBvecs(self.root, n=np.int64(13))
        for begin, end in ((-1, 1), (2, 1), (0, 14)):
            with self.assertRaises(IndexError):
                source.read(begin, end)

    def test_unused_tail_still_has_original_mapping_and_size_checks(self):
        source = DinoBvecs(self.root, n=4)
        self.assertEqual(len(source.maps), 2)
        self.assertEqual(source.ends[-1], 13)
        del source
        # Beyond-N files are not silently omitted from the original inventory.
        (self.chunks / "02.bvecs").write_bytes(b"\x05\x00")
        with self.assertRaisesRegex(ValueError, "incomplete"):
            DinoBvecs(self.root, n=4)

    def test_read_and_resident_staging_boundary(self):
        import torch

        rows = np.arange(65539 * 5, dtype=np.uint32).astype(np.uint8).reshape(-1, 5)
        self.write("00.bvecs", rows)
        source = DinoBvecs(self.root, n=len(rows) + 9)
        np.testing.assert_array_equal(source.read(0, len(rows)), rows)
        expected = np.concatenate((rows[-3:], self.data[4:7]))
        np.testing.assert_array_equal(source.read(len(rows) - 3, len(rows) + 3), expected)
        resident = source.resident(1, source.n - 1, "cpu")
        self.assertEqual(resident.dtype, torch.float32)
        self.assertEqual(resident.device.type, "cpu")
        expected = np.concatenate((rows[1:], self.data[4:-1])).astype(np.float32)
        np.testing.assert_array_equal(resident.numpy(), expected)

    def test_unfinished_download_rejects_required_prefix(self):
        marker = self.chunks / "01.bvecs.parts.json"
        marker.write_text("{}")
        # Preallocated size and valid headers do not establish completion.
        with self.assertRaisesRegex(ValueError, "unfinished"):
            DinoBvecs(self.root, n=13)
        with self.assertRaisesRegex(ValueError, "unfinished"):
            DinoBvecs(self.root, n=5)
        prefix = DinoBvecs(self.root, n=4)
        self.assertEqual(len(prefix.maps), 2)
        np.testing.assert_array_equal(prefix.read(0, 4), self.data[:4])
        marker.unlink()
        # Completion is a constructor gate; original mapped files stay immutable.
        (self.chunks / "00.bvecs.parts.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "unfinished"):
            DinoBvecs(self.root, n=4)

    def test_numbered_chunk_gap_is_not_skipped(self):
        (self.chunks / "00.bvecs").rename(self.chunks / "chunk_0000.bvecs")
        (self.chunks / "01.bvecs").rename(self.chunks / "chunk_0002.bvecs")
        with self.assertRaisesRegex(ValueError, "missing chunk"):
            DinoBvecs(self.root, n=13)
        with self.assertRaisesRegex(ValueError, "missing chunk"):
            DinoBvecs(self.root, n=5)
        source = DinoBvecs(self.root, n=4)
        np.testing.assert_array_equal(source.read(0, 4), self.data[:4])
        del source
        (self.chunks / "chunk_0000.bvecs").rename(self.chunks / "chunk_0001.bvecs")
        with self.assertRaisesRegex(ValueError, "missing chunk"):
            DinoBvecs(self.root, n=4)


if __name__ == "__main__":
    unittest.main()
