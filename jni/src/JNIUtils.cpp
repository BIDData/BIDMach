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

#include "Logger.hpp"
#include "JNIUtils.hpp"
#include "PointerUtils.hpp"

jmethodID String_getBytes; // ()[B


/**
 * Initialize the method IDs for the JNIUtils
 */
int initJNIUtils(JNIEnv *env)
{
    jclass cls = NULL;

    // Obtain the methodID for String#getBytes
    if (!init(env, cls, "java/lang/String")) return JNI_ERR;
    if (!init(env, cls, String_getBytes, "getBytes", "()[B")) return JNI_ERR;

    return JNI_VERSION_1_4;
}

/**
 * Initialize the given jclass, and return whether
 * the initialization succeeded
 */
bool init(JNIEnv *env, jclass& cls, const char *name)
{
    cls = env->FindClass(name);
    if (cls == NULL)
    {
        Logger::log(LOG_ERROR, "Failed to access class '%s'\n", name);
        return false;
    }
    return true;
}

/**
 * Initialize the specified field ID, and return whether
 * the initialization succeeded
 */
bool init(JNIEnv *env, jclass cls, jfieldID& field, const char *name, const char *signature)
{
    field = env->GetFieldID(cls, name, signature);
    if (field == NULL)
    {
        Logger::log(LOG_ERROR, "Failed to access field '%s'\n", name);
        return false;
    }
    return true;
}

/**
 * Initialize the specified method ID, and return whether
 * the initialization succeeded
 */
bool init(JNIEnv *env, jclass cls, jmethodID& method, const char *name, const char *signature)
{
    method = env->GetMethodID(cls, name, signature);
    if (method == NULL)
    {
        Logger::log(LOG_ERROR, "Failed to access method '%s'\n", name);
        return false;
    }
    return true;
}

/**
 * Initialize the given field ID with the field named 'nativePointer'
 * in the class with the given name
 */
bool initNativePointer(JNIEnv *env, jfieldID& field, const char *className)
{
    jclass cls = env->FindClass(className);
    if (cls == NULL)
    {
        Logger::log(LOG_ERROR, "Failed to access class %s\n", className);
        return false;
    }
    if (!init(env, cls, field, "nativePointer", "J")) return false;
    return true;
}




/**
 * Throws a new Java Exception that is identified by the given name, e.g.
 * "java/lang/IllegalArgumentException"
 * and contains the given message.
 */
void ThrowByName(JNIEnv *env, const char *name, const char *msg)
{
    jclass cls = env->FindClass(name);
    if (cls != NULL)
    {
        env->ThrowNew(cls, msg);
    }
    env->DeleteLocalRef(cls);
}



/**
 * Set the element at the given index in the given array to
 * the given value. If the array is NULL, nothing is done.
 * Returns 'false' if an OutOfMemoryError occurred or an
 * ArrayIndexOutOfBoundsExcepton was caused.
 */
bool set(JNIEnv *env, jintArray ja, int index, jint value)
{
    if (ja == NULL)
    {
        return true;
    }
    jsize len = env->GetArrayLength(ja);
    if (index < 0 || index >= len)
	{
        ThrowByName(env, "java/lang/ArrayIndexOutOfBoundsException",
            "Array index out of bounds");
		return false;
	}
    jint *a = (jint*)env->GetPrimitiveArrayCritical(ja, NULL);
    if (a == NULL)
    {
        return false;
    }
    a[index] = value;
    env->ReleasePrimitiveArrayCritical(ja, a, 0);
    return true;
}

/**
 * Set the element at the given index in the given array to
 * the given value. If the array is NULL, nothing is done.
 * Returns 'false' if an OutOfMemoryError occurred or an
 * ArrayIndexOutOfBoundsExcepton was caused.
 */
bool set(JNIEnv *env, jlongArray ja, int index, jlong value)
{
    if (ja == NULL)
    {
        return true;
    }
    jsize len = env->GetArrayLength(ja);
    if (index < 0 || index >= len)
	{
        ThrowByName(env, "java/lang/ArrayIndexOutOfBoundsException",
            "Array index out of bounds");
		return false;
	}
    jlong *a = (jlong*)env->GetPrimitiveArrayCritical(ja, NULL);
    if (a == NULL)
    {
        return false;
    }
    a[index] = value;
    env->ReleasePrimitiveArrayCritical(ja, a, 0);
    return true;
}




/**
 * Set the element at the given index in the given array to
 * the given value. If the array is NULL, nothing is done.
 * Returns 'false' if an OutOfMemoryError occurred or an
 * ArrayIndexOutOfBoundsExcepton was caused.
 */
bool set(JNIEnv *env, jfloatArray ja, int index, jfloat value)
{
    if (ja == NULL)
    {
        return true;
    }
    jsize len = env->GetArrayLength(ja);
    if (index < 0 || index >= len)
	{
        ThrowByName(env, "java/lang/ArrayIndexOutOfBoundsException",
            "Array index out of bounds");
		return false;
	}
    jfloat *a = (jfloat*)env->GetPrimitiveArrayCritical(ja, NULL);
    if (a == NULL)
    {
        return false;
    }
    a[index] = value;
    env->ReleasePrimitiveArrayCritical(ja, a, 0);
    return true;
}

/**
 * Set the element at the given index in the given array to
 * the given value. If the array is NULL, nothing is done.
 * Returns 'false' if an OutOfMemoryError occurred or an
 * ArrayIndexOutOfBoundsExcepton was caused.
 */
bool set(JNIEnv *env, jdoubleArray ja, int index, jdouble value)
{
    if (ja == NULL)
    {
        return true;
    }
    jsize len = env->GetArrayLength(ja);
    if (index < 0 || index >= len)
	{
        ThrowByName(env, "java/lang/ArrayIndexOutOfBoundsException",
            "Array index out of bounds");
		return false;
	}
    jdouble *a = (jdouble*)env->GetPrimitiveArrayCritical(ja, NULL);
    if (a == NULL)
    {
        return false;
    }
    a[index] = value;
    env->ReleasePrimitiveArrayCritical(ja, a, 0);
    return true;
}


/**
 * Returns the contents of the given array as a newly allocated
 * array, or NULL if the given array is NULL or any error occurs.
 * Deleting the returned array is left to the caller. The optional
 * 'length' argument will store the length of the given array.
 */
char* getArrayContents(JNIEnv *env, jbyteArray ja, int* length)
{
    if (ja == NULL)
    {
        return NULL;
    }
    jsize len = env->GetArrayLength(ja);
    if (length != NULL)
    {
        *length = (int)len;
    }
    jbyte *a = (jbyte*)env->GetPrimitiveArrayCritical(ja, NULL);
    if (a == NULL)
    {
        return NULL;
    }
    char *result = new char[len];
    if (result == NULL)
    {
        env->ReleasePrimitiveArrayCritical(ja, a, JNI_ABORT);
        return NULL;
    }
    for (int i=0; i<len; i++)
    {
        result[i] = (char)a[i];
    }
    env->ReleasePrimitiveArrayCritical(ja, a, JNI_ABORT);
    return result;
}


/**
 * Returns the contents of the given array as a newly allocated
 * array, or NULL if the given array is NULL or any error occurs.
 * Deleting the returned array is left to the caller. The optional
 * 'length' argument will store the length of the given array.
 */
int* getArrayContents(JNIEnv *env, jintArray ja, int* length)
{
    if (ja == NULL)
    {
        return NULL;
    }
    jsize len = env->GetArrayLength(ja);
    if (length != NULL)
    {
        *length = (int)len;
    }
    jint *a = (jint*)env->GetPrimitiveArrayCritical(ja, NULL);
    if (a == NULL)
    {
        return NULL;
    }
    int *result = new int[len];
    if (result == NULL)
    {
        env->ReleasePrimitiveArrayCritical(ja, a, JNI_ABORT);
        return NULL;
    }
    for (int i=0; i<len; i++)
    {
        result[i] = (int)a[i];
    }
    env->ReleasePrimitiveArrayCritical(ja, a, JNI_ABORT);
    return result;
}



/**
 * Converts the given jstring into a string and writes
 * the result into *s.
 *
 * Returns false iff the conversion process failed with
 * an out-of-memory-error
 */
/*
bool convertString(JNIEnv *env, jstring js, std::string *s)
{
    const char *str = env->GetStringUTFChars(js, NULL);
    if (str == NULL)
    {
        Logger::log(LOG_ERROR, "Out of memory during string creation\n");
        return false;
    }
    *s = str;
    env->ReleaseStringUTFChars(js, str);
    return true;
}
*/

/**
 * Converts the given jstring into a 0-terminated char* and
 * returns it. To delete the char* is left to the caller.
 * The optional length pointer will store the length of
 * the converted string, WITHOUT the trailing 0. Returns
 * NULL if an arror occurs.
 */
char *convertString(JNIEnv *env, jstring js, int *length)
{
    jbyteArray bytes = 0;
    char *result = 0;
    if (env->EnsureLocalCapacity(2) < 0)
    {
        ThrowByName(env, "java/lang/OutOfMemoryError",
            "Out of memory during string reference creation");
        return NULL;
    }
    bytes = (jbyteArray)env->CallObjectMethod(js, String_getBytes);
    if (!env->ExceptionCheck())
    {
        jint len = env->GetArrayLength(bytes);
        if (length != NULL)
        {
            *length = (int)len;
        }
        result = new char[len + 1];
        if (result == NULL)
        {
            ThrowByName(env, "java/lang/OutOfMemoryError",
                "Out of memory during string creation");
            return NULL;
        }
        env->GetByteArrayRegion(bytes, 0, len, (jbyte *)result);
        result[len] = 0;
    }
    return result;
}

/**
 * Returns the result of calling 'toString' on the given object.
 */
/*
std::string getToString(JNIEnv *env, jobject object)
{
	jclass cls = env->GetObjectClass(object);
    jmethodID mid = env->GetMethodID(cls, "toString", "()Ljava/lang/String;");
    if (mid == NULL)
	{
		Logger::log(LOG_ERROR, "Failed to access method 'toString'\n");
		return "[ERROR]";
	}
	if (object == NULL)
	{
		return "null";
	}
    jstring s = (jstring)env->CallObjectMethod(object, mid);
    const char *c = env->GetStringUTFChars(s, NULL);
	std::string result = "";
	result += c;
	env->ReleaseStringUTFChars(s, c);
	return result;
}
*/