package org.tensorflow.io;
import java.io.*;
import java.util.zip.*;

public class RecordWriter {
    private static final long serialVersionUID = 0L;
    private static final int DEFAULT_BUFSIZE = 64*1024;
    
    private BufferedOutputStream ds_;
    
    public RecordWriter(OutputStream ds) {
        ds_ = new BufferedOutputStream(ds, DEFAULT_BUFSIZE);
    }

    public RecordWriter(String fname) throws IOException {
        FileOutputStream fout = new FileOutputStream(fname);
        ds_ = new BufferedOutputStream(fout, DEFAULT_BUFSIZE);
    }

    public int maskedCRC(byte [] bytes, int count) {
        return CRC32C.mask(CRC32C.getValue(bytes, 0, count));
    }

    public int writeRecord(byte [] data) throws IOException {
        byte [] header = new byte[12];
        byte [] footer = new byte[4];
        CRC32C.encodeFixed64(header, 0, data.length);
        CRC32C.encodeFixed32(header, 8, maskedCRC(header, 8));

        CRC32C.encodeFixed32(footer, 0, maskedCRC(data, data.length));

        ds_.write(header, 0, 12);
        ds_.write(data, 0, data.length);
        ds_.write(footer, 0, 4);

        return 0;        
    }
}
