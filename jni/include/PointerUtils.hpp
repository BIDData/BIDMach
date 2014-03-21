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

#ifndef POINTERUTILS
#define POINTERUTILS

#include "JNIUtils.hpp"

class PointerData;



PointerData* initPointerData(JNIEnv *env, jobject nativePointerObject);
bool releasePointerData(JNIEnv *env, PointerData* &pointerData, jint mode=0);

void setNativePointerValue(JNIEnv *env, jobject nativePointerObject, jlong pointer);
void* getNativePointerValue(JNIEnv *env, jobject nativePointerObject);

void setPointer(JNIEnv *env, jobject pointerObject, jlong pointer);
void* getPointer(JNIEnv *env, jobject pointerObject);

bool isDirectByteBuffer(JNIEnv *env, jobject object);

bool isPointerBackedByNativeMemory(JNIEnv *env, jobject object);

int initPointerUtils(JNIEnv *env);

extern jmethodID Buffer_isDirect; // ()Z
extern jmethodID Buffer_hasArray; // ()Z
extern jmethodID Buffer_array; // ()Ljava/lang/Object;

extern jfieldID NativePointerObject_nativePointer; // long

extern jclass Pointer_class;
extern jfieldID Pointer_buffer; // Ljava.nio.Buffer;
extern jfieldID Pointer_pointers; // [jcuda.Pointer;
extern jfieldID Pointer_byteOffset; // long

extern jmethodID Object_getClass; // ()Ljava/lang/Class;

extern jmethodID Class_getComponentType; // ()Ljava/lang/Class;
extern jmethodID Class_newInstance; // ()Ljava/lang/Object;


/**
 * Virtual base class for all possible representations of pointers.
 */
class PointerData
{
    public:

        /**
         * Initialize this PointerData with the given object
         */
        virtual bool init(JNIEnv *env, jobject object) = 0;

        /**
         * Release this PointerData. To be called immediately
         * before the destructor call.
         */
        virtual bool release(JNIEnv *env, jint mode=0) = 0;



        /**
         * Returns the actual pointer represented by this PointerData
         */
        virtual void* getPointer(JNIEnv *env) = 0;

        /**
         * Releases the actual pointer. This should be called as soon
         * as possible after the pointer obtained with 'getPointer'
         * is no longer used, but it can be assumed that it is
         * called in the 'release' method, if necessary.
         */
        virtual void releasePointer(JNIEnv *env, jint mode=0) = 0;

        /**
         * When this pointer is an element of an array of pointers
         * that is pointed to by another pointer, then the contents
         * of this array may be modified by a CUDA function or
         * kernel. In this case, the updated native pointer value
         * will have to be written back into the Java NativePointerObject.
         * This will only work for the NativePointerData and
         * NativePointerObjectPointerData implementation. The method
         * returns whether the update succeeded. Otherwise, an
         * IllegalArgumentException will be thrown.
         */
        virtual bool setNewNativePointerValue(JNIEnv *env, jlong nativePointerValue) = 0;
};


// TODO The PointerData handling should be cleaned up:
// - Consider allowing setNewNativePointerValue to be called
//   on ALL implementations. Primarily, the array/buffer
//   references that are stored in the PointerData object
//   have to be cleared for that when the method is called,
//   but this has to be considered in the 'release' method
// - Consider creating Pointer instances when the
//   PointersArrayPointerData is released and points
//   to an array containing 'null' entries, but is
//   about to commit non-NULL local pointer values

/**
 * A PointerData that is backed by a Java NativePointerObject
 * (specifically, one that is not a Java Pointer). It only
 * stores the nativePointer value from the Java object.
 */
class NativePointerObjectPointerData : public PointerData
{
    private:

        /** The global reference to the Java NativePointerObject */
        jobject nativePointerObject;

        /** The nativePointer value from the Java NativePointerObject */
        jlong nativePointer;

    public:
        NativePointerObjectPointerData()
        {
            nativePointer = 0;
        }
        ~NativePointerObjectPointerData()
        {
        }

        bool init(JNIEnv *env, jobject object)
        {
            if (object != NULL)
            {
                // Create a global reference to the given object
                nativePointerObject = env->NewGlobalRef(object);
                if (nativePointerObject == NULL)
                {
                    ThrowByName(env, "java/lang/OutOfMemoryError",
                        "Out of memory while creating global reference for pointer data");
                    return false;
                }

                // Obtain the nativePointer value
                nativePointer = env->GetLongField(object, NativePointerObject_nativePointer);
                if (env->ExceptionCheck())
                {
                    return false;
                }
            }
            Logger::log(LOG_DEBUGTRACE, "Initialized  NativePointerObjectPointerData %p\n", nativePointer);
            return true;
        }

        bool release(JNIEnv *env, jint mode=0)
        {
            Logger::log(LOG_DEBUGTRACE, "Releasing    NativePointerObjectPointerData %p\n", nativePointer);
            env->SetLongField(nativePointerObject, NativePointerObject_nativePointer, nativePointer);
            env->DeleteGlobalRef(nativePointerObject);
            return true;
        }

        void* getPointer(JNIEnv *env)
        {
            return (void*)nativePointer;
        }

        void releasePointer(JNIEnv *env, jint mode=0)
        {
        }

        bool setNewNativePointerValue(JNIEnv *env, jlong nativePointerValue)
        {
            nativePointer = nativePointerValue;
            return true;
        }

};


/**
 * A PointerData that is backed by a Java Pointer. It stores the
 * nativePointer value from the NativePointerObject class, and
 * the byteOffset from the Pointer class.
 */
class NativePointerData : public PointerData
{
    private:

        /** The global reference to the Java Pointer */
        jobject pointer;

        /** The nativePointer value from the Java NativePointerObject */
        jlong nativePointer;

        /** The byteOffset from the Java Pointer */
        jlong byteOffset;

    public:
        NativePointerData()
        {
            nativePointer = 0;
            byteOffset = 0;
        }
        ~NativePointerData()
        {
        }

        bool init(JNIEnv *env, jobject object)
        {
            // Create a global reference to the given object
            pointer = env->NewGlobalRef(object);
            if (pointer == NULL)
            {
                ThrowByName(env, "java/lang/OutOfMemoryError",
                    "Out of memory while creating global reference for pointer data");
                return false;
            }

            // Obtain the nativePointer value
            nativePointer = env->GetLongField(object, NativePointerObject_nativePointer);
            if (env->ExceptionCheck())
            {
                return false;
            }

            // Obtain the byteOffset
            byteOffset = env->GetLongField(object, Pointer_byteOffset);
            if (env->ExceptionCheck())
            {
                return false;
            }

            Logger::log(LOG_DEBUGTRACE, "Initialized  NativePointerData              %p\n", nativePointer);
            return true;
        }

        bool release(JNIEnv *env, jint mode=0)
        {
            Logger::log(LOG_DEBUGTRACE, "Releasing    NativePointerData              %p\n", nativePointer);
            env->SetLongField(pointer, NativePointerObject_nativePointer, nativePointer);
            env->SetLongField(pointer, Pointer_byteOffset, byteOffset);
            env->DeleteGlobalRef(pointer);
            return true;
        }

        void* getPointer(JNIEnv *env)
        {
            return (void*)(((char*)nativePointer)+byteOffset);
        }

        void releasePointer(JNIEnv *env, jint mode=0)
        {
        }

        bool setNewNativePointerValue(JNIEnv *env, jlong nativePointerValue)
        {
            nativePointer = nativePointerValue;
            byteOffset = 0;
            return true;
        }

};


/**
 * A PointerData for a Java Pointer that points to an array
 * of NativePointerObjects. Internally it maintains an array
 * if PointerData objects, one for each NativePointerObject
 * of the array in the Java Pointer.
 */
class PointersArrayPointerData : public PointerData
{
    private:

        /** The global reference to the Java NativePointerObject */
        jobject nativePointerObject;

        /**
         * The array of PointerDatas, one for each of the Java
         * NativePointerObjects from the array inside the Java
         * Pointer
         */
        PointerData **arrayPointerDatas;

        /**
         * A pointer to a memory region that contains the
         * actual Pointer values from the NativePointerObjects
         */
        void *startPointer;

        /** The byteOffset from the Java Pointer */
        jlong byteOffset;


    public:

        PointersArrayPointerData()
        {
            arrayPointerDatas = NULL;
            startPointer = NULL;
            byteOffset = 0;
        }
        ~PointersArrayPointerData()
        {
        }

        bool init(JNIEnv *env, jobject object)
        {
            // Create a global reference to the given object
            nativePointerObject = env->NewGlobalRef(object);
            if (nativePointerObject == NULL)
            {
                ThrowByName(env, "java/lang/OutOfMemoryError",
                    "Out of memory while creating global reference for pointer data");
                return false;
            }

            jobjectArray pointersArray = (jobjectArray)env->GetObjectField(
                object, Pointer_pointers);
            long size = (long)env->GetArrayLength(pointersArray);

            // Prepare the pointer that points to the pointer
            // values of the NativePointerObjects
            void **localPointer = new void*[size];
            if (localPointer == NULL)
            {
                ThrowByName(env, "java/lang/OutOfMemoryError",
                    "Out of memory while initializing pointer array");
                return false;
            }
            startPointer = (void*)localPointer;

            // Prepare the PointerData objects for the Java NativePointerObjects
            arrayPointerDatas = new PointerData*[size];
            if (arrayPointerDatas == NULL)
            {
                ThrowByName(env, "java/lang/OutOfMemoryError",
                    "Out of memory while initializing pointer data array");
                return false;
            }

            // Initialize the PointerDatas and the pointer values
            // from the NativePointerObjects in the Java Pointer.
            for (int i=0; i<size; i++)
            {
                jobject p = env->GetObjectArrayElement(pointersArray, i);
                if (env->ExceptionCheck())
                {
                    return false;
                }
                if (p != NULL)
                {
                    // Initialize a PointerData for the pointer object that
                    // the pointer points to
                    PointerData *arrayPointerData = initPointerData(env, p);
                    if (arrayPointerData == NULL)
                    {
                        return false;
                    }
                    arrayPointerDatas[i] = arrayPointerData;
                    localPointer[i] = arrayPointerData->getPointer(env);
                }
                else
                {
                    arrayPointerDatas[i] = NULL;
                    localPointer[i] = NULL;
                }
            }

            // Obtain the byteOffset
            byteOffset = env->GetLongField(object, Pointer_byteOffset);
            if (env->ExceptionCheck())
            {
                return false;
            }

            Logger::log(LOG_DEBUGTRACE, "Initialized  PointersArrayPointerData       %p\n", startPointer);
            return true;
        }

        bool release(JNIEnv *env, jint mode=0)
        {
            Logger::log(LOG_DEBUGTRACE, "Releasing    PointersArrayPointerData       %p\n", startPointer);

            jobjectArray pointersArray = (jobjectArray)env->GetObjectField(
                nativePointerObject, Pointer_pointers);
            long size = (long)env->GetArrayLength(pointersArray);

            void **localPointer = (void**)startPointer;
            if (mode != JNI_ABORT)
            {
                // Write back the values from the native pointers array
                // into the Java objects
                for (int i=0; i<size; i++)
                {
                    jobject p = env->GetObjectArrayElement(pointersArray, i);
                    if (env->ExceptionCheck())
                    {
                        return false;
                    }
                    if (p != NULL)
                    {
                        // Check whether the value inside the pointer array has changed.
                        void *oldLocalPointer = arrayPointerDatas[i]->getPointer(env);

                        Logger::log(LOG_DEBUGTRACE, "About to write back pointer %d in PointersArrayPointerData\n", i);
                        Logger::log(LOG_DEBUGTRACE, "Old local pointer was %p\n", oldLocalPointer);
                        Logger::log(LOG_DEBUGTRACE, "New local pointer is  %p\n", localPointer[i]);

                        if (localPointer[i] != oldLocalPointer)
                        {
                            Logger::log(LOG_DEBUGTRACE, "In pointer %d setting value %p\n", i, localPointer[i]);
                            bool pointerUpdated = arrayPointerDatas[i]->setNewNativePointerValue(env, (jlong)localPointer[i]);
                            if (!pointerUpdated)
                            {
                                // If the pointer value could not be updated,
                                // (see setNewNativePointerValue documentation)
                                // then there is a pending IllegalArgumentException
                                return false;
                            }
                        }
                    }
                    else if (localPointer[i] != NULL)
                    {
                        // TODO: In future versions, it might be necessary to instantiate
                        // a pointer object here
                        ThrowByName(env, "java/lang/NullPointerException",
                            "Pointer points to an array containing a 'null' entry");
                        return false;
                    }
                }
            }

            // Release the PointerDatas for the pointer objects that
            // the pointer points to
            if (arrayPointerDatas != NULL)
            {
                for (int i=0; i<size; i++)
                {
                    if (arrayPointerDatas[i] != NULL)
                    {
                        if (!releasePointerData(env, arrayPointerDatas[i], mode)) return false;
                    }
                }
                delete[] arrayPointerDatas;
            }
            delete[] localPointer;

            env->DeleteGlobalRef(nativePointerObject);
            return true;
        }

        void* getPointer(JNIEnv *env)
        {
            return (void*)(((char*)startPointer)+byteOffset);
        }

        void releasePointer(JNIEnv *env, jint mode=0)
        {
        }

        bool setNewNativePointerValue(JNIEnv *env, jlong nativePointerValue)
        {
            ThrowByName(env, "java/lang/IllegalArgumentException",
                "Pointer to an array of pointers may not be overwritten");
            return false;
        }


};



/**
 * A PointerData that is backed by a direct Java Buffer
 */
class DirectBufferPointerData : public PointerData
{
    private:

        /** The address obtained from the direct buffer */
        void *startPointer;

        /** The byteOffset from the Java Pointer */
        jlong byteOffset;

    public:

        DirectBufferPointerData()
        {
            startPointer = NULL;
            byteOffset = 0;
        }
        ~DirectBufferPointerData()
        {
        }

        bool init(JNIEnv *env, jobject object)
        {
            // Obtain the direct buffer address from the given buffer
            jobject buffer = env->GetObjectField(object, Pointer_buffer);
            startPointer = env->GetDirectBufferAddress(buffer);
            if (startPointer == 0)
            {
                ThrowByName(env, "java/lang/IllegalArgumentException",
                    "Failed to obtain direct buffer address");
                return false;
            }

            // Obtain the byteOffset
            byteOffset = env->GetLongField(object, Pointer_byteOffset);
            if (env->ExceptionCheck())
            {
                return false;
            }

            Logger::log(LOG_DEBUGTRACE, "Initialized  DirectBufferPointerData        %p\n", startPointer);
            return true;
        }

        bool release(JNIEnv *env, jint mode=0)
        {
            Logger::log(LOG_DEBUGTRACE, "Releasing    DirectBufferPointerData        %p\n", startPointer);
            return true;
        }

        void* getPointer(JNIEnv *env)
        {
            return (void*)(((char*)startPointer)+byteOffset);
        }

        void releasePointer(JNIEnv *env, jint mode=0)
        {
        }

        bool setNewNativePointerValue(JNIEnv *env, jlong nativePointerValue)
        {
            ThrowByName(env, "java/lang/IllegalArgumentException",
                "Pointer to a direct buffer may not be overwritten");
            return false;
        }

};


/**
 * A PointerData that points to a Java Array
 */
class ArrayBufferPointerData : public PointerData
{
    private:

        /** A global reference to the Java array */
        jarray array;

        /** The address obtained from the Java array in 'getPointer' */
        void *startPointer;

        /** Whether the array was copied (and not pinned) */
        jboolean isCopy;

        /** The byteOffset from the Java Pointer */
        jlong byteOffset;


    public:

        ArrayBufferPointerData()
        {
            startPointer = NULL;
            array = NULL;
            byteOffset = 0;
            isCopy = JNI_FALSE;
        }
        ~ArrayBufferPointerData()
        {
        }

        bool init(JNIEnv *env, jobject object)
        {
            // Obtain the array reference
            jobject buffer = env->GetObjectField(object, Pointer_buffer);
            jobject localArray = env->CallObjectMethod(buffer, Buffer_array);
            if (env->ExceptionCheck())
            {
                return false;
            }
            array = (jarray)env->NewGlobalRef(localArray);
            if (array == NULL)
            {
                ThrowByName(env, "java/lang/OutOfMemoryError",
                    "Out of memory while creating array reference");
                return false;
            }

            // Obtain the byteOffset
            byteOffset = env->GetLongField(object, Pointer_byteOffset);
            if (env->ExceptionCheck())
            {
                return false;
            }

            Logger::log(LOG_DEBUGTRACE, "Initialized  ArrayBufferPointerData         %p (initialization is deferred)\n", startPointer);
            return true;
        }


        bool release(JNIEnv *env, jint mode=0)
        {
            Logger::log(LOG_DEBUGTRACE, "Releasing    ArrayBufferPointerData         %p\n", startPointer);
            releasePointer(env, mode);
            env->DeleteGlobalRef(array);
            return true;
        }

        void* getPointer(JNIEnv *env)
        {
            if (startPointer == NULL)
            {
                isCopy = JNI_FALSE;
                startPointer = env->GetPrimitiveArrayCritical(array, &isCopy);
                if (env->ExceptionCheck())
                {
                    return NULL;
                }
                Logger::log(LOG_DEBUGTRACE, "Initialized  ArrayBufferPointerData         %p (finished initialization, isCopy %d)\n", startPointer, isCopy);
            }
            return (void*)(((char*)startPointer)+byteOffset);
        }

        void releasePointer(JNIEnv *env, jint mode=0)
        {
            if (startPointer != NULL)
            {
                if (!isCopy)
                {
                    env->ReleasePrimitiveArrayCritical(array, startPointer, JNI_ABORT);
                }
                else
                {
                    env->ReleasePrimitiveArrayCritical(array, startPointer, mode);
                }
                startPointer = NULL;
            }
        }

        bool setNewNativePointerValue(JNIEnv *env, jlong nativePointerValue)
        {
            ThrowByName(env, "java/lang/IllegalArgumentException",
                "Pointer to an array may not be overwritten");
            return false;
        }

};



#endif