import hashlib
import importlib
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
packer = importlib.import_module('package_rental_results')


class ArchiveFastGzipTests(unittest.TestCase):
    def test_level_zero_retains_gzip_catalog_and_stream_integrity(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            output = project / 'delivery/project-results'
            output.mkdir(parents=True)
            source = project / 'runs/sample/data.bin'
            source.parent.mkdir(parents=True)
            payload = bytes(range(256)) * 128
            source.write_bytes(payload)
            relative = source.relative_to(project).as_posix()
            row = {'path': relative, 'group': 'runs/sample', 'stat': packer.metadata(source)}
            state = {'source_bytes_read': 0, 'rate': 10**9, 'gzip_level': 0}
            proof = packer.pack_chunk(project, output, 'runs/sample', [row], 'fast', 'gzip', state, lambda: None)
            archive = output / proof['archive']
            self.assertEqual(proof['compression'], 'gzip_level_0_single_thread')
            self.assertEqual(proof['sha256'], hashlib.sha256(archive.read_bytes()).hexdigest())
            self.assertEqual(proof['members'][0]['sha256_of_archived_file_bytes'], hashlib.sha256(payload).hexdigest())
            with tarfile.open(archive, 'r:gz') as stream:
                self.assertEqual(stream.extractfile(relative).read(), payload)
            catalog, archives = packer.load_catalog(output)
            self.assertEqual(catalog[relative], row['stat'])
            self.assertEqual(len(archives), 1)
            self.assertEqual(source.read_bytes(), payload)


if __name__ == '__main__':
    unittest.main()
