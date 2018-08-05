package org.tensorflow.io;
import java.io.*;
import java.util.zip.*;

public class RecordReader {
    private static final long serialVersionUID = 0L;
    private static final int DEFAULT_BUFSIZE = 64*1024;

    private BufferedInputStream ds_;
    
    public RecordReader(InputStream ds) {
        ds_ = new BufferedInputStream(ds, DEFAULT_BUFSIZE);
    }

    public RecordReader(String fname) throws IOException {
        FileInputStream fin = new FileInputStream(fname);
        ds_ = new BufferedInputStream(fin, DEFAULT_BUFSIZE);
    }

    public static int unmaskedCRC(byte [] bytes, int count) {
        return CRC32C.getValue(bytes, 0, count);
    }

    public byte [] readChecksummed(int n) throws IOException {

        int expected = n + 4;
        byte [] data = new byte[n];
        byte [] checksum = new byte[4];

        int actually_read = ds_.read(data, 0, n);
        actually_read += ds_.read(checksum, 0, 4);

        if (actually_read != expected) {
            throw new RuntimeException(String.format("RecordReader:readCheckSummed expected %d, read %d", expected, actually_read));
        }
        int masked_crc = CRC32C.decodeFixed32(checksum, 0);
        if (CRC32C.unmask(masked_crc) != unmaskedCRC(data, n)) {
            throw new RuntimeException("RecordReader:readCheckSummed checksum failed");
        }
        return data;
    }

    public byte [] readRaw(int n) throws IOException {

        byte [] data = new byte[n];

        int actually_read = ds_.read(data, 0, n);

        if (actually_read != n) {
            throw new RuntimeException(String.format("RecordReader:readRaw expected %d, read %d", n, actually_read));
        }

        return data;
    }


    public byte [] readRecord() throws IOException {
        byte [] reclenstr = readChecksummed(8);
        long reclen = CRC32C.decodeFixed64(reclenstr, 0);

        byte [] data = readChecksummed((int)reclen);
        return data;
    }

    public int available() throws IOException {
        return ds_.available();
    }
}
