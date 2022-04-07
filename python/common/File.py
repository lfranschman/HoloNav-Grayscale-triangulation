import os
import pickle
import gzip
try:
    import bz2
    G_BZIP2_LOADED = True
except ImportError:
    G_BZIP2_LOADED = False

try:
    from fs.zipfs import ZipFS
    from fs.tarfs import TarFS
    G_ZIP_LOADED = True
except ImportError:
    G_ZIP_LOADED = False

def get_filename_extension(path):
    return os.path.splitext(os.path.basename(path))[1]

if G_ZIP_LOADED:
    class ZipFileManager():
        ZIP_FS = 0
        ZIP_COUNT = 1

        def __init__(self):
            # print("ZipFileManager::__init__")
            self.m_ZipFileList = {}
            # self.m_pythonVersion = sys.version_info

        def __enter__(self):
            # print("ZipFileManager::__enter__")
            return

        def __exit__(self, exception_type, exception_value, traceback):
            # print("ZipFileManager::__exit__")
            # if sys.version_info < (3,0): # does not work all the time because sys may have been deleted before
            # if self.m_pythonVersion < (3,0):
                # for key, value in self.m_ZipFileList.iteritems():
                    # if value != None:
                        # print("warning closing unclosed zip file " + key)
                        # value[ZipFileManager.ZIP_COUNT].close()
            # else:
            for key, value in self.m_ZipFileList.items():
                if value is not None:
                    print("warning closing unclosed zip file " + key)
                    value[ZipFileManager.ZIP_FS].close()
            self.m_ZipFileList.clear()

        def __del__(self):
            # print("ZipFileManager::__del__")
            self.__exit__(exception_type=None, exception_value=None, traceback=None)

        def get_zip_file(self, zipfile):
            # print("ZipFileManager::get_zip_file")
            projectsFS = self.m_ZipFileList.get(zipfile)
            if projectsFS is None:
                print("ZipFileManager::get_zip_file " + str(zipfile))
                # projectsFS = ZipFS(zipfile, mode = 'r')
                fs = ZipFS(zipfile)
                self.m_ZipFileList[zipfile] = [fs, 0]
                projectsFS = self.m_ZipFileList[zipfile]

            projectsFS[ZipFileManager.ZIP_COUNT] += 1
            return projectsFS[ZipFileManager.ZIP_FS]

        def close_zip_file(self, zipfile):
            # print("ZipFileManager::close_zip_file")
            projectsFS = self.m_ZipFileList.get(zipfile)
            if projectsFS is not None:
                projectsFS[ZipFileManager.ZIP_COUNT] -= 1
                # if projectsFS[ZipFileManager.ZIP_COUNT] == 0:
                #     print("ZipFileManager::close_zip_file " + str(zipfile))
                #     projectsFS.close()
                #     del self.m_ZipFileList[zipfile]
            else:
                assert(False)

    gZipFileManager = ZipFileManager()

    class TarFileManager():
        TAR_FS = 0
        TAR_COUNT = 1

        def __init__(self):
            # print("TarFileManager::__init__")
            self.m_TarFileList = {}
            # self.m_pythonVersion = sys.version_info

        def __enter__(self):
            # print("TarFileManager::__enter__")
            return

        def __exit__(self, exception_type, exception_value, traceback):
            # print("TarFileManager::__exit__")
            # if sys.version_info < (3,0): # does not work all the time because sys may have been deleted before
            # if self.m_pythonVersion < (3,0):
            #     for key, value in self.m_TarFileList.iteritems():
            #         if value != None:
            #             print("warning closing unclosed tar file " + key)
            #             value[TarFileManager.TAR_COUNT].close()
            # else:
            for key, value in self.m_TarFileList.items():
                if value is not None:
                    print("warning closing unclosed tar file " + key)
                    value[TarFileManager.TAR_FS].close()
            self.m_TarFileList.clear()

        def __del__(self):
            # print("TarFileManager::__del__")
            self.__exit__(exception_type=None, exception_value=None, traceback=None)

        def get_tar_file(self, tarfile):
            # print("TarFileManager::get_tar_file")
            projectsFS = self.m_TarFileList.get(tarfile)
            if projectsFS is None:
                # print("TarFileManager::get_tar_file " + str(tarfile))
                # projectsFS = TarFS(tarfile, mode = 'r')
                fs = TarFS(tarfile)
                self.m_TarFileList[tarfile] = [fs, 0]
                projectsFS = self.m_TarFileList[tarfile]

            projectsFS[TarFileManager.TAR_COUNT] += 1
            return projectsFS[TarFileManager.TAR_FS]

        def close_tar_file(self, tarfile):
            # print("TarFileManager::close_tar_file")
            projectsFS = self.m_TarFileList.get(tarfile)
            if projectsFS is not None:
                projectsFS[TarFileManager.TAR_COUNT] -= 1
                # if projectsFS[TarFileManager.TAR_COUNT] == 0:
                #     print("TarFileManager::close_tar_file " + str(tarfile))
                #     projectsFS.close()
                #     del self.m_TarFileList[tarfile]
            else:
                assert(False)

    gTarFileManager = TarFileManager()

def open_file(filename, flags):
    if G_ZIP_LOADED:
        # if the file is inside a zip file "//C:/blabla.zip//insideZip/lala.txt"
        if filename[0] == '/' and filename[1] == '/':
            pos = filename.find("//", 2)
            zipFile = filename[2:pos]
            tmp_filename = filename[pos + 2:]
            realFilename = zipFile[:-4] + '/' + tmp_filename

            # check first if the real file exist before looking in the zip file
            if flags == 'w' or os.path.isfile(realFilename):
                return open(realFilename, flags)

            global gZipFileManager
            projectsFS = gZipFileManager.get_zip_file(zipFile)
            return projectsFS.open(tmp_filename)

    return open(filename, flags)

def is_file_exist(fileName):
    if G_ZIP_LOADED:
        # if the file is inside a zip or tar file "//C:/blabla.zip//insideZip/lala.txt"
        if fileName[0] == '/' and fileName[1] == '/':
            pos = fileName.find("//", 2)
            archiveFile = fileName[2:pos]
            extension = fileName[pos - 3:pos]
            filename = fileName[pos + 2:]
            realFilename = archiveFile[:-4] + '/' + filename
            # check first if the real file exist before looking in the archive
            if os.path.isfile(realFilename):
                return True

            if os.path.isfile(archiveFile):
                if extension == "zip":
                    global gZipFileManager
                    projectsFS = gZipFileManager.get_zip_file(archiveFile)
                    exist = projectsFS.isfile(filename)
                    gZipFileManager.close_zip_file(archiveFile)
                else:
                    global gTarFileManager
                    projectsFS = gTarFileManager.get_tar_file(archiveFile)
                    exist = projectsFS.isfile(filename)
                    gTarFileManager.close_tar_file(archiveFile)
                return exist

            return False

    return os.path.isfile(fileName)

# if _path finish with '.gz' the file will be compressed with gzip
# if _path finish with '.bz2' the file will be compressed with bzip2
def save_pickle(var, path):
    if get_filename_extension(path) == '.gz':
        with gzip.GzipFile(path, 'w') as f:
            pickle.dump(var, f, -1)
    elif get_filename_extension(path) == '.bz2':
        with bz2.BZ2File(path, 'w') as f:
            pickle.dump(var, f, -1)
    else:
        with open_file(path, 'wb') as f:
            pickle.dump(var, f, -1)

# if _path finish with '.gz' the file will be decompressed with gzip
# if _path finish with '.bz2' the file will be decompressed with bzip2
def load_pickle(path):
    if get_filename_extension(path) == '.gz':
        with gzip.open(path, 'rb') as f:
            var = pickle.load(f)
    elif get_filename_extension(path) == '.bz2':
        with bz2.open(path, 'rb') as f:
            var = pickle.load(f)
    else:
        with open_file(path, 'rb') as f:
            var = pickle.load(f)
    return var
