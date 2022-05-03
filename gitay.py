import argparse
import difflib
from genericpath import isdir
from gettext import find
import hashlib
import logging
import os
import time

import zlib
from re import S, finditer, compile
from collections import OrderedDict
from typing import List, Set, Union
from pathlib import Path
from configparser import ConfigParser

logging.basicConfig(level=logging.DEBUG)


def cmd_init(args):
    repo = GitayRepository(args.path)
    repo.create_repo()


def cmd_cat(args):
    gitay_object = GitayObject(find_repo(), args.object).read()
    print(gitay_object.serialize())


def cmd_hash(args):
    raw_data = Path(args.path).read_bytes()

    gitay_object = FORMATS_OBJECTS[args.type](raw_data)
    sha = gitay_object.write(args.write)
    print(sha)


def cmd_log(args):
    commit_object = GitayObject(find_repo(), args.commit).read()
    print(commit_object.log_commit(commits=set()))


def cmd_ls(args):
    gitay_object = GitayObject(find_repo(), args.object).read()
    tree_object = gitay_object.find_tree()
    print(tree_object.print())


def cmd_checkout(args):
    repo = find_repo()
    gitay_object = GitayObject(repo, args.commit).read()
    gitay_object = gitay_object.find_tree()
    if gitay_object.file_format == b'commit':
        # get tree object
        tree = gitay_object.commit_dict.get(b'tree', b'')[0].decode()
        gitay_object: GitayTree = GitayObject(
            repo, repo.get_object_hash(tree)).read()

    assert gitay_object.file_format == b'tree'

    path = Path(args.path)
    if path.exists():
        if not path.is_dir():
            # todo change
            raise Exception('path is not dir')
        elif list(path.iterdir):
            raise Exception('path is note empty')
    else:
        path.mkdir(parents=True, exist_ok=True)

    gitay_object.checkout(path)


def cmd_show_refs(args):
    repo = find_repo()
    refs = repo.get_refs()
    for path, sha in refs.items():
        print(f"{sha} {path}")


def cmd_tag(args):
    if args.name:
        tag = GitayTag(find_repo(), "")
        tag.create(args.name, args.object, args.create_tag_object)
    else:
        repo = find_repo()
        refs = repo.get_refs('tags')
        for path, sha in refs.items():
            print(f"{sha} {path}")


def cmd_rev(args):
    repo = find_repo()
    print(repo.get_object_hash(args.name))


def cmd_ls_files(args):
    index_file = GitayIndex()
    for entry in index_file.entries:
        if args.stage:
            mask_flags = 0b00110000000000000
            print(
                f"{entry.mode:6o} {entry.sha} {entry.flags & mask_flags} {entry.path}")
        else:
            print(entry.path)


def cmd_status(args):
    index_file = GitayIndex()
    index_paths = {i.path: i.sha for i in index_file.entries}
    files = {p.as_posix() for p in Path().glob("**/*") if p.is_file()
             and '.git/' not in p.as_posix()}
    changed_files = set()
    for path in index_paths.keys() & files:
        sha = hashlib.sha1(Path(path).read_bytes()).hexdigest()
        if sha != index_paths[path]:
            changed_files.add(path)

    print("changed files:")
    for p in changed_files:
        print(p)
    print("\n\nnew files:")
    for p in files - index_paths.keys():
        print(p)
    print("\n\nremoved files:")
    for p in index_paths.keys() - files:
        print(p)


def cmd_diff(args):
    repo = find_repo()
    index_file = GitayIndex()
    index_paths = {i.path: i.sha for i in index_file.entries}
    files = {p.as_posix() for p in Path().glob("**/*") if p.is_file()
             and '.git/' not in p.as_posix()}
    for path in index_paths.keys() & files:
        file_bytes = Path(path).read_bytes()
        sha = hashlib.sha1(file_bytes).hexdigest()
        if sha != index_paths[path]:
            index_object = GitayObject(repo, index_paths[path]).read()
            index_lines = index_object.serialize().decode().splitlines()
            working_lines = file_bytes.decode().splitlines()
            diff_lines = difflib.unified_diff(
                index_lines, working_lines, "index", "working")
            print('\n'.join(diff_lines))


def cmd_add(args):
    index_file = GitayIndex()
    index_file.add_entry(args.path)
    index_file.write()
    print(f"added {args.path}")

def cmd_create_tree(args):
    create_tree(find_repo(), Path("./.gitignore").read_text().splitlines(), Path())

def cmd_commit(args):
    repo = find_repo()
    index_list = [node.path for node in GitayIndex().entries]
    curr_tree = create_tree(repo, Path("./.gitignore").read_text().splitlines(), Path(), index_list)
    commit_object = GitayCommit(repo)
    commit_object.commit_dict[b'tree'] = [curr_tree.encode()]
    head = Path(repo.get_repo_file_path('HEAD').read_text().removeprefix('ref: ').removesuffix('\n'))
    if head.is_file():
        # todo fid get_object_hash
        commit_object.commit_dict[b'parent'] = [repo.get_object_hash('HEAD').encode()]
    
    timestamp = int(time.mktime(time.localtime()))
    utc_offset = -time.timezone
    author_time = '{} {}{:02}{:02}'.format(
            timestamp,
            '+' if utc_offset > 0 else '-',
            abs(utc_offset) // 3600,
            (abs(utc_offset) // 60) % 60)
    commit_object.commit_dict[b'author'] = [b'itay-ye <itay.yeshaya@gmail.com> ' + author_time.encode()]
    commit_object.commit_dict[b'committer'] = [b'itay-ye <itay.yeshaya@gmail.com> ' + author_time.encode()]
    commit_object.message = args.message.encode()
    sha = commit_object.write(True)
    path = repo.get_repo_file_path('HEAD')
    data = path.read_text()
    ref_path = repo.get_repo_file_path(data.removeprefix('ref: ').removesuffix('\n'), True)
    ref_path.write_text(sha)


def create_blob_object(file_path: Path):
    assert file_path.is_file()
    blob_object = GitayBlob(find_repo(), data=file_path.read_bytes())
    res = blob_object.write(save_file=True)
    print(res)


class GitayRepository:
    """Repository class"""

    def __init__(self, path: str = ".") -> None:
        self.worktree = Path(path)
        self.gitdir = self.worktree / ".git"

    def create_repo(self):
        self.dir(self.worktree, True)
        self.dir(self.gitdir, True)

        self.dir(self.gitdir / "objects", True)
        self.dir(self.gitdir / "refs", True)
        self.dir(self.gitdir / "refs" / "tags", True)
        self.dir(self.gitdir / "refs" / "heads", True)
        self.dir(self.gitdir / "branches", True)
        (self.gitdir / "description").write_text("This is an empty description\n")
        (self.gitdir / "HEAD").write_text("ref: refs/heads/master\n")
        self.create_new_conf()

    @staticmethod
    def dir(path: Union[Path, str], mkdir: bool = False) -> None:
        if isinstance(path, str):
            path = Path(path)

        if not path.is_dir():
            if mkdir:
                logging.debug(f"creating {path}")
                path.mkdir(parents=True, exist_ok=True)
            else:
                raise GitayRepository.InvalidPath(path.absolute())
        else:
            logging.debug(f"{path} exists")

    def get_repo_file_path(self, file_path: Union[Path, str], force=False) -> Path:
        file_path = self.gitdir / file_path
        if not force and not file_path.exists():
            raise self.InvalidPath(file_path.absolute())

        return file_path

    def get_conf(self) -> ConfigParser:
        """

        Returns:
            ConfigParser: [description]
        """
        parser = ConfigParser()
        parser.read(self.gitdir / 'config')
        return parser

    def create_new_conf(self):
        temp_parser = ConfigParser()
        temp_parser.add_section("core")
        temp_parser.set("core", "repositoryformatversion", "0")
        temp_parser.set("core", "filemode", "false")
        temp_parser.set("core", "bare", "false")
        temp_parser.write((self.gitdir / 'config').open('w'))

    def get_refs(self, types="") -> OrderedDict:
        refs_dir = self.gitdir / 'refs' / types
        all_files = [path for path in refs_dir.glob(
            '**/*') if not path.is_dir()]
        all_files.sort()
        result_dict = OrderedDict()
        for path in all_files:
            data = path.read_text()
            if data.startswith('ref: '):
                ref_path = Path(data.removeprefix('ref: ').removesuffix('\n'))
                assert ref_path in all_files
            else:
                result_dict[str(Path(*path.parts[1:]))
                            ] = data.removesuffix('\n')
        return result_dict

    def get_single_ref(self, ref_path):
        path = self.get_repo_file_path(ref_path)
        data = path.read_text()
        if data.startswith('ref: '):
            ref_path = Path(data.removeprefix('ref: ').removesuffix('\n'))
            return self.get_single_ref(ref_path)
        else:
            return data.removesuffix('\n')

    def get_object_hash(self, name):
        hash_regex = compile(r"^[0-9A-Fa-f]{4,40}$")
        if not name:
            return None

        if name == 'HEAD':
            return self.get_single_ref('HEAD')

        tags: Path = self.gitdir / 'refs' / 'tags'
        for tag in tags.iterdir():
            if tag.name == name:
                return self.get_single_ref(tag)

        if hash_regex.match(name):
            if len(name) == 40:
                # this is full hash
                return name
            else:
                dir_path: Path = self.gitdir / 'objects' / name[:2]
                if dir_path.is_dir():
                    files = [f for f in dir_path.iterdir(
                    ) if f.name.startswith(name[2:])]
                    if len(files) > 1:
                        raise Exception(
                            f"Ambiguous name {name} - options = {[name[:2] + file for file in files]}")
                    return files[0]

    class InvalidPath(Exception):
        def __init__(self, path, *args, **kwargs):
            msg = f"{path} is not a valid path."
            super().__init__(msg, *args, **kwargs)

    class MissingRepository(Exception):
        def __init__(self, *args, **kwargs):
            msg = "No repository found."
            super().__init__(msg, *args, **kwargs)


def create_tree(repo: GitayRepository, git_ignore_files: List, root_path: Path = Path(), index_list=list()):
    print(f"create tree for {root_path}: ")
    tree = GitayTree(repo)
    for p in root_path.iterdir():
        print(p)
        if p.as_posix() in git_ignore_files or p.as_posix().startswith('.'):
            continue

        if p.is_file() and p.as_posix() in index_list:
            blob_object = GitayBlob(repo, data=p.read_bytes())
            sha = blob_object.write(save_file=True)
            mode = oct(p.stat().st_mode)[2:]

        elif p.is_dir():
            sha = create_tree(repo, git_ignore_files, p, index_list)
            if not sha:
                continue
            new_path = repo.get_repo_file_path(f'objects/{sha[:2]}/{sha[2:]}')
            mode = oct(new_path.stat().st_mode)[2:]
        else:
            continue

        tree.add_node(mode.encode(), p.as_posix().encode(), bytes.fromhex(sha))

    if tree.nodes:
        return tree.write(True)

    return ""


class GitayObject:
    def __init__(self, repo: GitayRepository, name: str, data: str = "") -> None:
        if data:
            self.deserialize(data)
        if name:
            self.sha = repo.get_object_hash(name)
        self.repo = repo

    def read(self):
        object_path = Path("objects", self.sha[0:2], self.sha[2:])
        file_path = self.repo.get_repo_file_path(object_path)
        file_bytes = file_path.read_bytes()
        raw_data = zlib.decompress(file_bytes)
        file_format, raw_data = raw_data.split(b' ', 1)
        file_format = file_format.decode('ascii')
        file_size, raw_data = raw_data.split(b'\x00', 1)
        file_size = int(file_size.decode('ascii'))
        if file_size != len(raw_data):
            raise GitayObject.CorruptFile(object_path)

        gitay_object = FORMATS_OBJECTS[file_format](
            self.repo, self.sha, raw_data)
        return gitay_object

    def serialize(self):
        pass

    def deserialize(self, data: bytes):
        pass

    class CorruptFile(Exception):
        def __init__(self, path, *args, **kwargs):
            msg = f"{path} Corrupt file."
            super().__init__(msg, *args, **kwargs)

    def write(self, save_file=False):
        data = self.serialize()
        raw_data = self.file_format + b' ' + \
            str(len(data)).encode() + b'\x00' + data
        sha = hashlib.sha1(raw_data).hexdigest()

        if save_file:
            # Compute path
            object_dir_path = Path("objects", sha[0:2])
            self.repo.dir(self.repo.gitdir / object_dir_path, mkdir=True)
            file_path = self.repo.get_repo_file_path(
                object_dir_path / sha[2:], force=True)
            file_path.write_bytes(zlib.compress(raw_data))
            print(f"saved file {self.file_format} in {file_path}")
        return sha

    def find_tree(self):
        if self.file_format == b'tree':
            return self
        elif self.file_format == b'blob':
            # should never get here
            return None
        elif self.file_format == b'commit':
            return GitayObject(self.repo, self.commit_dict[b'tree'][0].decode()).read().find_tree()
        elif self.file_format == b'tag':
            GitayObject(
                self.repo, self.commit_dict[b'object'][0].decode()).read().find_tree()


class GitayBlob(GitayObject):
    file_format = b'blob'

    def __init__(self, repo: GitayRepository, sha: str = "", data: str = "") -> None:
        super().__init__(repo, sha, data)

    def serialize(self):
        if not self.data:
            return b""

        return self.data

    def deserialize(self, data: bytes):
        self.data = data


class GitayCommit(GitayObject):
    file_format = b'commit'

    def __init__(self, repo: GitayRepository, sha: str = "", data: str = "") -> None:
        super().__init__(repo, sha, data)
        self.commit_dict = OrderedDict()

    def serialize(self) -> bytes:
        result_string = b""
        for k, v in self.commit_dict.items():
            for element in v:
                element = element.replace(b'\n', b'\n ')
                result_string += k + b' ' + element + b'\n'

        result_string += b'\n' + self.message

        return result_string

    def deserialize(self, data: bytes):
        self.commit_dict = OrderedDict()
        end = 0

        for entry in finditer(rb"(([\S]+( [\S]+)+)(\n( [\S]+))?)", data):
            end = entry.span()[-1]
            value = entry.group()
            key, value = value.split(b' ', 1)
            # value = value.replace(b' ', b'')
            value = self.commit_dict.get(key, []) + [value]
            self.commit_dict[key] = value

        self.message = data[end:]

    def log_commit(self, commits: Set):
        if self.sha in commits:
            return
        commits.add(self.sha)
        buffer = ""
        if b'parent' not in self.commit_dict:
            buffer += "Initial commit!\n"
        buffer += f"commit {self.sha}\n"
        buffer += f"{self.serialize().decode()}"

        for parent in self.commit_dict.get(b'parent', []):
            commit_object = GitayObject(find_repo(), parent.decode()).read()
            buffer += commit_object.log_commit(commits)

        return buffer

    def get_tree(self):
        return self.commit_dict[b'tree']


class GitayTree(GitayObject):
    file_format = b'tree'

    def __init__(self, repo: GitayRepository, sha: str = "", data: str = "") -> None:
        super().__init__(repo, sha, data)
        self.nodes = []

    def serialize(self) -> bytes:
        self.nodes.sort(key=lambda x: x.path)
        return b''.join([node.serialize() for node in self.nodes])

    def deserialize(self, data: bytes):
        self.nodes = []
        while data:
            mode, data = data.split(b' ', 1)
            path, data = data.split(b'\x00', 1)
            sha, data = data[:20], data[20:]
            self.nodes.append(self.Node(mode, path, sha))

    def print(self):
        return b'\n'.join([node.print() for node in self.nodes]).decode()

    def checkout(self, dest: Path):
        for node in self.nodes:
            gitay_object = GitayObject(find_repo(), node.sha).read()
            path = Path(node.path.decode())
            path = dest / path
            if gitay_object.file_format == b'tree':
                path.resolve()
                gitay_object.checkout(dest)

            elif gitay_object.file_format == b'blob':
                path.write_bytes(gitay_object.serialize())

    class Node:
        def __init__(self, mode, path, sha) -> None:
            self.mode = mode
            self.path = path
            self.sha = sha.hex()

        def serialize(self):
            res = self.mode + b' ' + self.path + \
                b'\x00' + bytes.fromhex(self.sha)
            return res

        def print(self):
            object_format = GitayObject(
                find_repo(), self.sha).read().file_format
            res = self.mode + b' ' + object_format + b' ' + self.path + \
                b' ' + self.sha.encode()
            return res

    def add_node(self, mode, path, sha) -> None:
        node = self.Node(mode, path, sha)
        self.nodes.append(node)


class GitayTag(GitayCommit):
    file_format = b'tag'

    def create(self, name, dest, heavy=False):
        if heavy:
            result_dict = OrderedDict()
            result_dict[b'object'] = [dest.encode()]
            result_dict[b'type'] = [b'commit']
            result_dict[b'tag'] = [name.encode()]
            result_dict[b'tagger'] = [b'Gitay']
            self.message = b"This is a message!"
            self.commit_dict = result_dict
            self.data = self.serialize()
            dest = self.write(True)

        tags_path: Path = find_repo().gitdir / "refs" / "tags" / name
        tags_path.write_text(dest)


class GitayIndex:
    class IndexEntry:
        def __init__(self, data=b"", **kwargs) -> None:
            if not data:
                self.init(**kwargs)
            else:
                self.data = data
                self.nbytes = 0

                self.ctime_s = self._read_int()
                self.ctime_n = self._read_int()
                self.mtime_s = self._read_int()
                self.mtime_n = self._read_int()
                self.dev = self._read_int()
                self.ino = self._read_int()
                """
                32-bit mode, split into (high to low bits)

                    4-bit object type
                    valid values in binary are 1000 (regular file), 1010 (symbolic link)
                    and 1110 (gitlink)

                    3-bit unused

                    9-bit unix permission. Only 0755 and 0644 are valid for regular files.
                    Symbolic links and gitlinks have value 0 in this field.
                """
                self.mode = self._read_int()
                self.uid = self._read_int()
                self.gid = self._read_int()
                self.file_size = self._read_int()
                self.sha = self._read_sha()
                self.flags = self._read_flags()
                self.path = self._read_path()

                del self.data

        def init(self, ctime_s, ctime_n, mtime_s, mtime_n, dev, ino, mode, uid, gid, file_size, sha, flags, path):
            self.ctime_s = ctime_s
            self.ctime_n = ctime_n
            self.mtime_s = mtime_s
            self.mtime_n = mtime_n
            self.dev = dev
            self.ino = ino
            self.mode = mode
            self.uid = uid
            self.gid = gid
            self.file_size = file_size
            self.sha = sha
            self.flags = flags
            self.path = path
            return self

        def _read_int(self):
            self.nbytes += 4
            res = int.from_bytes(self.data[:4], "big")
            self.data = self.data[4:]
            return res

        def _read_sha(self):
            self.nbytes += 20
            res = self.data[:20].hex()
            self.data = self.data[20:]
            return res

        def _read_flags(self):
            self.nbytes += 2
            res = int.from_bytes(self.data[:2], "big")
            self.data = self.data[2:]
            return res

        def _read_path(self):
            path_end = self.data.find(b'\00')
            path = self.data[:path_end].decode()
            self.nbytes += path_end
            self.nbytes = (((self.nbytes) + (1 + 7)) // 8) * 8  # with pads
            return path

        def serialize(self):
            res = b''
            res += self.ctime_s.to_bytes(4, "big")
            res += self.ctime_n.to_bytes(4, "big")
            res += self.mtime_s.to_bytes(4, "big")
            res += self.mtime_n.to_bytes(4, "big")
            res += self.dev.to_bytes(4, "big")
            res += self.ino.to_bytes(4, "big")
            res += self.mode.to_bytes(4, "big")
            res += self.uid.to_bytes(4, "big")
            res += self.gid.to_bytes(4, "big")
            res += self.file_size.to_bytes(4, "big")
            res += bytes.fromhex(self.sha)
            res += self.flags.to_bytes(2, "big")
            res += self.path.encode() + b'\x00'
            if len(res) % 8:
                npads = ((((len(res)) + (7)) // 8) * 8) - len(res)
                res += b'\x00'*npads

            return res

    def __init__(self, path=None) -> None:
        self.entries = []
        repo = find_repo()
        if path:
            self.index_path = path
        else:
            self.index_path = repo.get_repo_file_path('index', True)
        if not self.index_path.is_file():
            return
        index_file = self.index_path.read_bytes()
        if not index_file:
            return
        # could use struct module here
        self.sha, index_file = index_file[-20:].hex(), index_file[:-20]
        assert self.sha == hashlib.sha1(index_file).hexdigest()

        self.signature, index_file = index_file[:4].decode(), index_file[4:]
        self.version, index_file = int.from_bytes(
            index_file[:4], "big"), index_file[4:]
        self.num_entries, index_file = int.from_bytes(
            index_file[:4], "big"), index_file[4:]

        assert self.signature == 'DIRC'
        assert self.version == 2

        self.entries = []
        while len(index_file) > 62:
            if index_file[:4] == b'TREE' or index_file[:4] == b"REUC":
                break
            r = self.IndexEntry(index_file)
            index_file = index_file[r.nbytes:]
            self.entries.append(r)

        assert len(self.entries) == self.num_entries

    def add_entry(self, path):
        path = Path(path)
        assert path.is_file()
        sha = hashlib.sha1(path.read_bytes()).hexdigest()
        file_stat = path.stat()
        entry = self.IndexEntry(
            ctime_s=int(file_stat.st_ctime),
            ctime_n=0,
            mtime_s=int(file_stat.st_mtime),
            mtime_n=0,
            dev=int(file_stat.st_dev),
            ino=int(file_stat.st_ino),
            mode=int(file_stat.st_mode),
            uid=int(file_stat.st_uid),
            gid=int(file_stat.st_gid),
            file_size=int(file_stat.st_size),
            sha=sha,
            flags=len(path.as_posix().encode()),
            path=path.as_posix()
        )
        if path not in [i.path for i in self.entries]:
            self.entries.append(entry)

    def write(self):
        self.entries.sort(key=lambda x: x.path)
        path = Path(self.index_path)
        result_string = b'DIRC'
        result_string += int('2').to_bytes(4, "big")
        result_string += len(self.entries).to_bytes(4, "big")
        for entry in self.entries:
            result_string += entry.serialize()
        result_string += hashlib.sha1(result_string).digest()
        path.write_bytes(result_string)


FORMATS_OBJECTS = {
    'blob': GitayBlob,
    'commit': GitayCommit,
    'tree': GitayTree,
    'tag': GitayTag
}


def find_repo(path=Path(),  root=Path().root) -> GitayRepository:
    git_dir = path / ".git"
    if git_dir.is_dir():
        return GitayRepository(path)

    if path == root:
        raise GitayRepository.MissingRepository()

    find_repo(path=path.parent, root=root)

def get_remote_hash():
    pass
def parse_arguments():
    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(metavar="COMMAND")
    subparser.required = True

    parser_init = subparser.add_parser("init")
    parser_init.add_argument("path", metavar="PATH", nargs="?",
                             default=".", help="Where to create the repository.")
    parser_init.set_defaults(func=cmd_init)

    parser_cat = subparser.add_parser("cat-file",
                                      help="Provide content of repository objects")
    parser_cat.add_argument("object", metavar="OBJECT",
                            help="The object to display.")
    parser_cat.set_defaults(func=cmd_cat)

    parser_hash = subparser.add_parser(
        "hash-object", help="Compute object ID and optionally creates a blob from a file")
    parser_hash.add_argument("-t", metavar="type", dest="type", choices=['blob', 'commit', 'tag', 'tree'],
                             help="Speicfy the file type.", default="blob",)
    parser_hash.add_argument("-w", dest="write",
                             action="store_true",
                             help="Save the file into the repository.")
    parser_hash.add_argument("path", metavar="PATH",
                             help="Read object from <file>")
    parser_hash.set_defaults(func=cmd_hash)

    parser_log = subparser.add_parser(
        "log", help="Display history of a given commit.")

    parser_log.add_argument("commit", default="HEAD",
                            nargs="?", help="Commit to start at.")
    parser_log.set_defaults(func=cmd_log)

    parser_ls = subparser.add_parser(
        "ls-tree", help="Pretty-print a tree object.")
    parser_ls.add_argument("object",
                           help="The object to show.")
    parser_ls.set_defaults(func=cmd_ls)

    parser_checkout = subparser.add_parser(
        "checkout", help="Checkout a commit inside of a directory.")

    parser_checkout.add_argument("commit",
                               help="The commit or tree to checkout.")

    parser_checkout.add_argument("path",
                               help="The EMPTY directory to checkout on.")
    parser_checkout.set_defaults(func=cmd_checkout)

    parser_show_refs = subparser.add_parser(
        "show-ref", help="List references.")
    parser_show_refs.set_defaults(func=cmd_show_refs)

    parser_tag = subparser.add_parser("tag", help="List and create tags")
    parser_tag.add_argument("-a",
                            action="store_true",
                            dest="create_tag_object",
                            help="Whether to create a tag object")
    parser_tag.add_argument("name",
                            nargs="?",
                            help="The new tag's name")
    parser_tag.add_argument("object",
                            default="HEAD",
                            nargs="?",
                            help="The object the new tag will point to")
    parser_tag.set_defaults(func=cmd_tag)

    parser_rev = subparser.add_parser("rev-parse",
                                      help="Parse revision (or other objects )identifiers")
    parser_rev.add_argument("name",
                            help="The name to parse")
    parser_rev.set_defaults(func=cmd_rev)

    lf_files_parser = subparser.add_parser("ls-files",
                                           help='list files in index')
    lf_files_parser.add_argument('-s', '--stage', action='store_true',
                                 help='show object details (mode, hash, and stage number) in '
                                 'addition to path')
    lf_files_parser.set_defaults(func=cmd_ls_files)

    status_parser = subparser.add_parser("status",
                                         help='current files status')
    status_parser.set_defaults(func=cmd_status)

    diff_parser = subparser.add_parser("diff",
                                       help='get diff of all files')
    diff_parser.set_defaults(func=cmd_diff)

    add_parser = subparser.add_parser("add",
                                      help='add file to index')
    add_parser.add_argument("path",
                            help="The path of the file to add")
    add_parser.set_defaults(func=cmd_add)

    create_tree_parser = subparser.add_parser("create-tree",
                                      help='create current tree')
    create_tree_parser.set_defaults(func=cmd_create_tree)

    commit_parser = subparser.add_parser("commit",
                                      help='create commit')
    commit_parser.add_argument("message",
                            help="Commit message")
    commit_parser.set_defaults(func=cmd_commit)



    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    parse_arguments()  # pragma: no cover
