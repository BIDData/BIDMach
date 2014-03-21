/*
 * JCuda - Java bindings for NVIDIA CUDA driver and runtime API
 *
 * Copyright (c) 2009-2012 Marco Hutter - http://www.jcuda.org
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use,
 * copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef JNIUTILS
#define JNIUTILS

#include <jni.h>
#include <string>

#ifdef _WIN32
    // Disable "unreferenced formal parameter" warnings
    #pragma warning (disable : 4100)
#endif


bool init(JNIEnv *env, jclass& cls, const char *name);
bool init(JNIEnv *env, jclass cls, jfieldID& field, const char *name, const char *signature);
bool init(JNIEnv *env, jclass cls, jmethodID& method, const char *name, const char *signature);
bool initNativePointer(JNIEnv *env, jfieldID& field, const char *className);

bool set(JNIEnv *env, jintArray ja, int index, jint value);
bool set(JNIEnv *env, jlongArray ja, int index, jlong value);
bool set(JNIEnv *env, jfloatArray ja, int index, jfloat value);
bool set(JNIEnv *env, jdoubleArray ja, int index, jdouble value);

int* getArrayContents(JNIEnv *env, jintArray ja, int* length=NULL);
char* getArrayContents(JNIEnv *env, jbyteArray ja, int* length=NULL);

//bool convertString(JNIEnv *env, jstring js, std::string *s);
char *convertString(JNIEnv *env, jstring js, int *length=NULL);

//std::string getToString(JNIEnv *env, jobject object);

void ThrowByName(JNIEnv *env, const char *name, const char *msg);

int initJNIUtils(JNIEnv *env);

extern jmethodID String_getBytes; // ()[B


#endif