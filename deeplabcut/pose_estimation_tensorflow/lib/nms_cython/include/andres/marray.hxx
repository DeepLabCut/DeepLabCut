/// \mainpage
/// Marray: Fast Runtime-Flexible Multi-dimensional Arrays and Views in C++.
/// \newline
///
/// Copyright (c) 2014 by Bjoern Andres, bjoern@andres.sc
///
/// \section section_abstract Short Description
/// Marray is a single header file for fast multi-dimensional arrays and views 
/// in C++. Unlike in other implementations such as boost MultiArray and 
/// Blitz++, the dimension of Marray views and arrays can be set and changed at 
/// runtime. Dimension is not a template parameter in Marray. Arrays and views 
/// that have the same type of entries but different dimension are therefore of 
/// the same C++ type. In conjunction with the comprehensive and 
/// convenient Marray interface, this brings some of the flexibility known from 
/// high-level languages such as Python, R and MATLAB to C++.
///
/// \section section_features Features
/// - Multi-dimensional arrays and views whose dimension, shape, size and 
///   indexing order (first or last coordinate major order) can be set and 
///   changed at runtime.
/// - Access to entries via coordinates, scalar indices, STL-compliant random 
///   access iterators and C++11 initializer lists.
/// - Arithmetic operators with expression templates and automatic type 
///   promotion.
/// - Support for STL-compliant allocators.
/// 
/// \section section_tutorial Tutorial
/// - An introductory tutorial can be found at src/tutorial/tutorial.cxx
///
/// \section section_cpp0x C++11 Extensions
/// - C++11 extensions are enabled by defining
///   - HAVE_CPP11_VARIADIC_TEMPLATES
///   - HAVE_CPP11_INITIALIZER_LISTS
///   - HAVE_CPP11_TEMPLATE_ALIASES
///   .
/// 
/// \section section_license License
/// Copyright (c) 2013 by Bjoern Andres.
/// 
/// This software was developed by Bjoern Andres.
/// Enquiries shall be directed to bjoern@andres.sc.
///
/// Redistribution and use in source and binary forms, with or without 
/// modification, are permitted provided that the following conditions are met:
/// - Redistributions of source code must retain the above copyright notice,
///   this list of conditions and the following disclaimer.
/// - Redistributions in binary form must reproduce the above copyright notice, 
///   this list of conditions and the following disclaimer in the documentation
///   and/or other materials provided with the distribution.
/// - The name of the author must not be used to endorse or promote products 
///   derived from this software without specific prior written permission.
///
/// THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR IMPLIED 
/// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF 
/// MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO 
/// EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
/// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
/// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; 
/// OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
/// WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR 
/// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF 
/// ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
/// 
#pragma once
#ifndef ANDRES_MARRAY_HXX
#define ANDRES_MARRAY_HXX

#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <limits>
#include <string>
#include <sstream>
#include <cstring> // memcpy
#include <iterator> 
#include <vector>
#include <set>
#include <iostream> 
#include <memory> // allocator
#include <numeric> // accumulate
#include <functional> // std::multiplies
#ifdef HAVE_CPP11_INITIALIZER_LISTS
    #include <initializer_list>
#endif

/// The public API.
namespace andres {

enum StringStyle {TableStyle, MatrixStyle}; ///< Flag to be used with the member function asString() of View.
enum CoordinateOrder {FirstMajorOrder, LastMajorOrder}; ///< Flag setting the order of coordinate tuples.
struct InitializationSkipping { }; ///< Flag to indicate initialization skipping.

static const bool Const = true; ///< Flag to be used with the template parameter isConst of View and Iterator.
static const bool Mutable = false; ///< Flag to be used with the template parameter isConst of View and Iterator.
static const CoordinateOrder defaultOrder = LastMajorOrder; ///< Default order of coordinate tuples.
static const InitializationSkipping SkipInitialization = InitializationSkipping(); ///< Flag to indicate initialization skipping.

template<class E, class T> 
    class ViewExpression;
// \cond suppress_doxygen
template<class E, class T, class UnaryFunctor> 
    class UnaryViewExpression;
template<class E1, class T1, class E2, class T2, class BinaryFunctor> 
    class BinaryViewExpression;
template<class E, class T, class S, class BinaryFunctor> 
    class BinaryViewExpressionScalarFirst;
template<class E, class T, class S, class BinaryFunctor> 
    class BinaryViewExpressionScalarSecond;
// \endcond suppress_doxygen
template<class T, bool isConst = false, class A = std::allocator<std::size_t> > 
    class View;
#ifdef HAVE_CPP11_TEMPLATE_ALIASES
    template<class T, class A> using ConstView = View<T, true, A>;
#endif
template<class T, bool isConst, class A = std::allocator<std::size_t> > 
    class Iterator;
template<class T, class A = std::allocator<std::size_t> > class Marray;

// assertion testing
#ifdef NDEBUG
    const bool MARRAY_NO_DEBUG = true; ///< General assertion testing disabled.
    const bool MARRAY_NO_ARG_TEST = true; ///< Argument testing disabled.
#else
    const bool MARRAY_NO_DEBUG = false; ///< General assertion testing enabled.
    const bool MARRAY_NO_ARG_TEST = false; ///< Argument testing enabled.
#endif

// \cond suppress_doxygen
namespace marray_detail {
    // meta-programming
    template <bool PREDICATE, class TRUECASE, class FALSECASE>
        struct IfBool;
    template <class TRUECASE, class FALSECASE>
        struct IfBool<true, TRUECASE, FALSECASE>
        { typedef TRUECASE type; };
    template <class TRUECASE, class FALSECASE>
        struct IfBool<false, TRUECASE, FALSECASE>
        { typedef FALSECASE type; };

    template <class T1, class T2>
        struct IsEqual
        { static const bool type = false; };
    template <class T>
        struct IsEqual<T, T>
        { static const bool type = true; };

    template<class T> struct TypeTraits
        { static const unsigned char position = 255; };
    template<> struct TypeTraits<char> 
        { static const unsigned char position = 0; };
    template<> struct TypeTraits<unsigned char> 
        { static const unsigned char position = 1; };
    template<> struct TypeTraits<short> 
        { static const unsigned char position = 2; };
    template<> struct TypeTraits<unsigned short> 
        { static const unsigned char position = 3; };
    template<> struct TypeTraits<int> 
        { static const unsigned char position = 4; };
    template<> struct TypeTraits<unsigned int> 
        { static const unsigned char position = 5; };
    template<> struct TypeTraits<long> 
        { static const unsigned char position = 6; };
    template<> struct TypeTraits<unsigned long> 
        { static const unsigned char position = 7; };
    template<> struct TypeTraits<float> 
        { static const unsigned char position = 8; };
    template<> struct TypeTraits<double> 
        { static const unsigned char position = 9; };
    template<> struct TypeTraits<long double> 
        { static const unsigned char position = 10; };
    template<class A, class B> struct PromoteType
        { typedef typename IfBool<TypeTraits<A>::position >= TypeTraits<B>::position, A, B>::type type; };

    // assertion testing
    template<class A> inline void Assert(A assertion) {
        if(!assertion) throw std::runtime_error("Assertion failed.");
    }

    // geometry of views
    template<class A = std::allocator<std::size_t> > class Geometry;
    template<class ShapeIterator, class StridesIterator>
        inline void stridesFromShape(ShapeIterator, ShapeIterator,
            StridesIterator, const CoordinateOrder& = defaultOrder);

    // operations on entries of views
    template<class Functor, class T, class A>
        inline void operate(View<T, false, A>&, Functor);
    template<class Functor, class T, class A>
        inline void operate(View<T, false, A>&, const T&, Functor);
    template<class Functor, class T1, class T2, bool isConst, class A>
        inline void operate(View<T1, false, A>&, const View<T2, isConst, A>&, Functor);
    template<class Functor, class T1, class A, class E, class T2>
        inline void operate(View<T1, false, A>& v, const ViewExpression<E, T2>& expression, Functor f);

    // helper classes 
    template<unsigned short N, class Functor, class T, class A>
        struct OperateHelperUnary;
    template<unsigned short N, class Functor, class T1, class T2, class A>
        struct OperateHelperBinaryScalar;
    template<unsigned short N, class Functor, class T1, class T2, 
             bool isConst, class A1, class A2>
        struct OperateHelperBinary;
    template<bool isConstTo, class TFrom, class TTo, class AFrom, class ATo> 
        struct AssignmentOperatorHelper;
    template<bool isIntegral> 
        struct AccessOperatorHelper;

    // unary in-place functors
    template<class T>
        struct Negative { void operator()(T& x) { x = -x; } };
    template<class T>
        struct PrefixIncrement { void operator()(T& x) { ++x; } };
    template<class T>
        struct PostfixIncrement { void operator()(T& x) { x++; } };
    template<class T>
        struct PrefixDecrement { void operator()(T& x) { --x; } };
    template<class T>
        struct PostfixDecrement { void operator()(T& x) { x--; } };

    // binary in-place functors
    template<class T1, class T2>
        struct Assign { void operator()(T1& x, const T2& y) { x = y; } };
    template<class T1, class T2>
        struct PlusEqual { void operator()(T1& x, const T2& y) { x += y; } };
    template<class T1, class T2>
        struct MinusEqual { void operator()(T1& x, const T2& y) { x -= y; } };
    template<class T1, class T2>
        struct TimesEqual { void operator()(T1& x, const T2& y) { x *= y; } };
    template<class T1, class T2>
        struct DividedByEqual { void operator()(T1& x, const T2& y) { x /= y; } };

    // unary functors
    template<class T>
        struct Negate { T operator()(const T& x) const { return -x; } };

    // binary functors
    template<class T1, class T2, class U>
        struct Plus { U operator()(const T1& x, const T2& y) const { return x + y; } };
    template<class T1, class T2, class U>
        struct Minus { U operator()(const T1& x, const T2& y) const { return x - y; } };
    template<class T1, class T2, class U>
        struct Times { U operator()(const T1& x, const T2& y) const { return x * y; } };
    template<class T1, class T2, class U>
        struct DividedBy { U operator()(const T1& x, const T2& y) const { return x / y; } };
}
// \endcond suppress_doxygen
   
/// Array-Interface to an interval of memory.
///
/// A view makes a subset of memory look as if it was stored in an 
/// Marray. With the help of a view, data in a subset of memory can 
/// be accessed and manipulated conveniently. In contrast to arrays
/// which allocate and de-allocate their own memory, views only 
/// reference memory that has been allocated by other means.
/// Perhaps the simplest and most important use of views is to
/// read and manipulate sub-arrays.
///
/// Notes on arithmetic operators of View:
/// - Only the pre-fix operators ++ and -- and not the corresponding post-fix
///   operators are implemented for View because the return value of the
///   post-fix operators would have to be the View as it is prior to the 
///   operator call. However, the data under the view cannot be preserved when 
///   incrementing or decrementing. Some compilers might accept the post-fix
///   operators, use the pre-fix implementation implicitly and issue a warning.
///
template<class T, bool isConst, class A> 
class View 
: public ViewExpression<View<T, isConst, A>, T>
{
public:
    typedef T value_type;
    typedef typename marray_detail::IfBool<isConst, const T*, T*>::type pointer;
    typedef const T* const_pointer;
    typedef typename marray_detail::IfBool<isConst, const T&, T&>::type reference;
    typedef const T& const_reference;
    typedef Iterator<T, isConst, A> iterator;
    typedef Iterator<T, true, A> const_iterator;
    typedef std::reverse_iterator<iterator> reverse_iterator;
    typedef std::reverse_iterator<const_iterator> const_reverse_iterator;
    typedef ViewExpression<View<T, isConst, A>, T> base;
    typedef typename A::template rebind<value_type>::other allocator_type;

    // construction
    View(const allocator_type& = allocator_type());
    View(pointer, const allocator_type& = allocator_type()); 
    View(const View<T, false, A>&);
    template<class ShapeIterator>
        View(ShapeIterator, ShapeIterator, pointer,
            const CoordinateOrder& = defaultOrder,
            const CoordinateOrder& = defaultOrder,
            const allocator_type& = allocator_type());
    template<class ShapeIterator, class StrideIterator>
        View(ShapeIterator, ShapeIterator, StrideIterator,
            pointer, const CoordinateOrder&, 
            const allocator_type& = allocator_type());
    #ifdef HAVE_CPP11_INITIALIZER_LISTS
        View(std::initializer_list<std::size_t>, pointer,
            const CoordinateOrder& = defaultOrder,
            const CoordinateOrder& = defaultOrder,
            const allocator_type& = allocator_type());
        View(std::initializer_list<std::size_t>, std::initializer_list<std::size_t>,
            pointer, const CoordinateOrder&, 
            const allocator_type& = allocator_type());
    #endif
    
    // assignment
    View<T, isConst, A>& operator=(const T&);
    View<T, isConst, A>& operator=(const View<T, true, A>&); // over-write default
    View<T, isConst, A>& operator=(const View<T, false, A>&); // over-write default
    template<class TLocal, bool isConstLocal, class ALocal>
        View<T, isConst, A>& operator=(const View<TLocal, isConstLocal, ALocal>&); 
    template<class E, class Te>
        View<T, isConst, A>& 
        operator=(const ViewExpression<E, Te>&);

    void assign(const allocator_type& = allocator_type());
    template<class ShapeIterator>
        void assign(ShapeIterator, ShapeIterator, pointer,
            const CoordinateOrder& = defaultOrder,
            const CoordinateOrder& = defaultOrder,
            const allocator_type& = allocator_type());
    template<class ShapeIterator, class StrideIterator>
        void assign(ShapeIterator, ShapeIterator, StrideIterator,
            pointer, const CoordinateOrder&, 
            const allocator_type& = allocator_type());
    #ifdef HAVE_CPP11_INITIALIZER_LISTS
        void assign(std::initializer_list<std::size_t>, pointer,
            const CoordinateOrder& = defaultOrder,
            const CoordinateOrder& = defaultOrder,
            const allocator_type& = allocator_type());
        void assign(std::initializer_list<std::size_t>, 
            std::initializer_list<std::size_t>, pointer,
            const CoordinateOrder&, 
            const allocator_type& = allocator_type());
    #endif
        
    // query
    const std::size_t dimension() const;
    const std::size_t size() const;
    const std::size_t shape(const std::size_t) const;
    const std::size_t* shapeBegin() const;
    const std::size_t* shapeEnd() const;
    const std::size_t strides(const std::size_t) const;
    const std::size_t* stridesBegin() const;
    const std::size_t* stridesEnd() const;
    const CoordinateOrder& coordinateOrder() const;
    const bool isSimple() const; 
    template<class TLocal, bool isConstLocal, class ALocal> 
        bool overlaps(const View<TLocal, isConstLocal, ALocal>&) const;

    // element access
    template<class U> reference operator()(U); 
    template<class U> reference operator()(U) const; 
    #ifndef HAVE_CPP11_VARIADIC_TEMPLATES
        reference operator()(const std::size_t, const std::size_t);
        reference operator()(const std::size_t, const std::size_t) const;
        reference operator()(const std::size_t, const std::size_t, const std::size_t);
        reference operator()(const std::size_t, const std::size_t, const std::size_t) const;
        reference operator()(const std::size_t, const std::size_t, const std::size_t, 
            const std::size_t);
        reference operator()(const std::size_t, const std::size_t, const std::size_t, 
            const std::size_t) const;
        reference operator()(const std::size_t, const std::size_t, const std::size_t, 
             const std::size_t, const std::size_t);
        reference operator()(const std::size_t, const std::size_t, const std::size_t, 
            const std::size_t, const std::size_t) const;
        reference operator()(const std::size_t, const std::size_t, const std::size_t, 
            const std::size_t, const std::size_t, const std::size_t, const std::size_t, 
            const std::size_t, const std::size_t, const std::size_t);
        reference operator()(const std::size_t, const std::size_t, const std::size_t, 
            const std::size_t, const std::size_t, const std::size_t, const std::size_t, 
            const std::size_t, const std::size_t, const std::size_t) const;
    #else
        reference operator()(const std::size_t);
        reference operator()(const std::size_t) const;
        template<typename... Args>
            reference operator()(const std::size_t, const Args...);
        template<typename... Args>
            reference operator()(const std::size_t, const Args...) const;
        private:
            std::size_t elementAccessHelper(const std::size_t, const std::size_t);
            std::size_t elementAccessHelper(const std::size_t, const std::size_t) const;
            template<typename... Args>
                std::size_t elementAccessHelper(const std::size_t, const std::size_t,
                    const Args...);
            template<typename... Args>
                std::size_t elementAccessHelper(const std::size_t, const std::size_t, 
                    const Args...) const;
        public:
    #endif

    // sub-views
    template<class BaseIterator, class ShapeIterator>
        void view(BaseIterator, ShapeIterator, View<T, isConst, A>&) const;
    template<class BaseIterator, class ShapeIterator>
        void view(BaseIterator, ShapeIterator, const CoordinateOrder&,
            View<T, isConst, A>&) const;
    template<class BaseIterator, class ShapeIterator>
        View<T, isConst, A> view(BaseIterator, ShapeIterator) const;
    template<class BaseIterator, class ShapeIterator>
        View<T, isConst, A> view(BaseIterator, ShapeIterator,
            const CoordinateOrder&) const;
    template<class BaseIterator, class ShapeIterator>
        void constView(BaseIterator, ShapeIterator, View<T, true, A>&) const;
    template<class BaseIterator, class ShapeIterator>
        void constView(BaseIterator, ShapeIterator, const CoordinateOrder&,
            View<T, true, A>&) const;
    template<class BaseIterator, class ShapeIterator>
        View<T, true, A> constView(BaseIterator, ShapeIterator) const;
    template<class BaseIterator, class ShapeIterator>
        View<T, true, A> constView(BaseIterator, ShapeIterator, 
            const CoordinateOrder&) const;
    #ifdef HAVE_CPP11_INITIALIZER_LISTS
        void view(std::initializer_list<std::size_t>,
            std::initializer_list<std::size_t>, View<T, isConst, A>&) const;
        void view(std::initializer_list<std::size_t>,
            std::initializer_list<std::size_t>, const CoordinateOrder&,
            View<T, isConst, A>&) const;
        void constView(std::initializer_list<std::size_t>,
            std::initializer_list<std::size_t>, View<T, true, A>&) const;
        void constView(std::initializer_list<std::size_t>,
            std::initializer_list<std::size_t>, const CoordinateOrder&,
            View<T, true, A>&) const;
    #endif

    // iterator access
    iterator begin();
    iterator end();
    const_iterator begin() const;
    const_iterator end() const;
    reverse_iterator rbegin();
    reverse_iterator rend();
    const_reverse_iterator rbegin() const;
    const_reverse_iterator rend() const;

    // coordinate transformation
    template<class ShapeIterator>
        void reshape(ShapeIterator, ShapeIterator);
    template<class CoordinateIterator>
        void permute(CoordinateIterator);
    void transpose(const std::size_t, const std::size_t);
    void transpose();
    void shift(const int);
    void squeeze();

    template<class ShapeIterator>
        View<T, isConst, A> reshapedView(ShapeIterator, ShapeIterator) const;
    template<class CoordinateIterator>
        View<T, isConst, A> permutedView(CoordinateIterator) const;
    View<T, isConst, A> transposedView(const std::size_t, const std::size_t) const;
    View<T, isConst, A> transposedView() const;
    View<T, isConst, A> shiftedView(const int) const;
    View<T, isConst, A> boundView(const std::size_t, const std::size_t = 0) const;
    View<T, isConst, A> squeezedView() const;

    #ifdef HAVE_CPP11_INITIALIZER_LISTS
        void reshape(std::initializer_list<std::size_t>);
        void permute(std::initializer_list<std::size_t>);

        View<T, isConst, A> reshapedView(std::initializer_list<std::size_t>) const;
        View<T, isConst, A> permutedView(std::initializer_list<std::size_t>) const;
    #endif

    // conversion between coordinates, index and offset
    template<class CoordinateIterator>
        void coordinatesToIndex(CoordinateIterator, std::size_t&) const;
    template<class CoordinateIterator>
        void coordinatesToOffset(CoordinateIterator, std::size_t&) const;
    template<class CoordinateIterator>
        void indexToCoordinates(std::size_t, CoordinateIterator) const;
    void indexToOffset(std::size_t, std::size_t&) const;
    #ifdef HAVE_CPP11_INITIALIZER_LISTS
        void coordinatesToIndex(std::initializer_list<std::size_t>,
            std::size_t&) const;
        void coordinatesToOffset(std::initializer_list<std::size_t>,
            std::size_t&) const;
    #endif

    // output as string
    std::string asString(const StringStyle& = MatrixStyle) const; 

private:
    typedef typename marray_detail::Geometry<A> geometry_type;

    View(pointer, const geometry_type&); 

    void updateSimplicity();
    void testInvariant() const; 

    // unsafe direct memory access
    const T& operator[](const std::size_t) const;
    T& operator[](const std::size_t);

    // data and memory address
    pointer data_;
    geometry_type geometry_;

template<class TLocal, bool isConstLocal, class ALocal>
    friend class View;
template<class TLocal, class ALocal>
    friend class Marray;
// \cond suppress_doxygen
template<bool isConstTo, class TFrom, class TTo, class AFrom, class ATo> 
    friend struct marray_detail::AssignmentOperatorHelper;
friend struct marray_detail::AccessOperatorHelper<true>;
friend struct marray_detail::AccessOperatorHelper<false>;

template<class E, class U> 
    friend class ViewExpression;
template<class E, class U, class UnaryFunctor> 
    friend class UnaryViewExpression;
template<class E1, class T1, class E2, class T2, class BinaryFunctor> 
    friend class BinaryViewExpression;
template<class E, class U, class S, class BinaryFunctor> 
    friend class BinaryViewExpressionScalarFirst;
template<class E, class U, class S, class BinaryFunctor> 
    friend class BinaryViewExpressionScalarSecond;

template<class Functor, class T1, class Alocal, class E, class T2>
    friend void marray_detail::operate(View<T1, false, Alocal>& v, const ViewExpression<E, T2>& expression, Functor f);
// \endcond end suppress_doxygen
};

/// STL-compliant random access iterator for View and Marray.
/// 
/// In addition to the STL iterator interface, the member functions
/// hasMore(), index(), and coordinate() are defined.
///
template<class T, bool isConst, class A>
class Iterator
{
public:
    // STL random access iterator typedefs
    typedef typename std::random_access_iterator_tag iterator_category;
    typedef T value_type;
    typedef std::ptrdiff_t difference_type;
    typedef typename marray_detail::IfBool<isConst, const T*, T*>::type pointer;
    typedef typename marray_detail::IfBool<isConst, const T&, T&>::type reference;

    // non-standard typedefs
    typedef typename marray_detail::IfBool<isConst, const View<T, true, A>*,
        View<T, false, A>*>::type view_pointer;
    typedef typename marray_detail::IfBool<isConst, const View<T, true, A>&,
        View<T, false, A>&>::type view_reference;

    // construction
    Iterator();
    Iterator(const View<T, false, A>&, const std::size_t = 0); 
    Iterator(View<T, false, A>&, const std::size_t = 0); 
    Iterator(const View<T, true, A>&, const std::size_t = 0);
    Iterator(const Iterator<T, false, A>&);
        // conversion from mutable to const

    // STL random access iterator operations
    reference operator*() const;
    pointer operator->() const;
    reference operator[](const std::size_t) const;
    Iterator<T, isConst, A>& operator+=(const difference_type&);
    Iterator<T, isConst, A>& operator-=(const difference_type&);
    Iterator<T, isConst, A>& operator++(); // prefix

    Iterator<T, isConst, A>& operator--(); // prefix
    Iterator<T, isConst, A> operator++(int); // postfix
    Iterator<T, isConst, A> operator--(int); // postfix
    Iterator<T, isConst, A> operator+(const difference_type&) const;
    Iterator<T, isConst, A> operator-(const difference_type&) const;
    template<bool isConstLocal>
        difference_type operator-(const Iterator<T, isConstLocal, A>&) const;
    template<bool isConstLocal>
        bool operator==(const Iterator<T, isConstLocal, A>&) const;
    template<bool isConstLocal>
        bool operator!=(const Iterator<T, isConstLocal, A>&) const;
    template<bool isConstLocal>
        bool operator<(const Iterator<T, isConstLocal, A>&) const;
    template<bool isConstLocal>
        bool operator>(const Iterator<T, isConstLocal, A>&) const;
    template<bool isConstLocal>
        bool operator<=(const Iterator<T, isConstLocal, A>&) const;
    template<bool isConstLocal>
        bool operator>=(const Iterator<T, isConstLocal, A>&) const;

    // operations beyond the STL standard
    bool hasMore() const; 
    std::size_t index() const;
    template<class CoordinateIterator>
        void coordinate(CoordinateIterator) const;

private:
    void testInvariant() const;

    // attributes
    view_pointer view_; 
    pointer pointer_;
    std::size_t index_;
    std::vector<std::size_t> coordinates_;

friend class Marray<T, A>;
friend class Iterator<T, !isConst, A>; // for comparison operators
};

/// Runtime-Flexible multi-dimensional array.
template<class T, class A> 
class Marray
: public View<T, false, A>
{
public:
    typedef View<T, false, A> base;
    typedef typename base::value_type value_type;
    typedef typename base::pointer pointer;
    typedef typename base::const_pointer const_pointer;
    typedef typename base::reference reference;
    typedef typename base::const_reference const_reference;
    typedef typename base::iterator iterator;
    typedef typename base::reverse_iterator reverse_iterator;
    typedef typename base::const_iterator const_iterator;
    typedef typename base::const_reverse_iterator const_reverse_iterator;
    typedef typename A::template rebind<value_type>::other allocator_type;

    // constructors and destructor
    Marray(const allocator_type& = allocator_type());
    Marray(const T&, const CoordinateOrder& = defaultOrder, 
        const allocator_type& = allocator_type());
    template<class ShapeIterator>
        Marray(ShapeIterator, ShapeIterator, const T& = T(),
            const CoordinateOrder& = defaultOrder, 
            const allocator_type& = allocator_type());
    template<class ShapeIterator>
        Marray(const InitializationSkipping&, ShapeIterator, ShapeIterator,
            const CoordinateOrder& = defaultOrder, 
            const allocator_type& = allocator_type());
    #ifdef HAVE_CPP11_INITIALIZER_LISTS
        Marray(std::initializer_list<std::size_t>, const T& = T(),
            const CoordinateOrder& = defaultOrder, 
            const allocator_type& = allocator_type());
    #endif
    Marray(const Marray<T, A>&);
    template<class E, class Te>
        Marray(const ViewExpression<E, Te>&,
            const allocator_type& = allocator_type());
    template<class TLocal, bool isConstLocal, class ALocal>
        Marray(const View<TLocal, isConstLocal, ALocal>&);
    ~Marray();
    
    // assignment
    Marray<T, A>& operator=(const T&);
    Marray<T, A>& operator=(const Marray<T, A>&); // over-write default
    template<class TLocal, bool isConstLocal, class ALocal>
        Marray<T, A>& operator=(const View<TLocal, isConstLocal, ALocal>&);
    template<class E, class Te>
        Marray<T, A>& operator=(const ViewExpression<E, Te>&);
    void assign(const allocator_type& = allocator_type());

    // resize
    template<class ShapeIterator>
        void resize(ShapeIterator, ShapeIterator, const T& = T());
    template<class ShapeIterator>
        void resize(const InitializationSkipping&, ShapeIterator, ShapeIterator);
    #ifdef HAVE_CPP11_INITIALIZER_LISTS
        void resize(std::initializer_list<std::size_t>, const T& = T());
        void resize(const InitializationSkipping&, std::initializer_list<std::size_t>);
    #endif

private:
    typedef typename base::geometry_type geometry_type;

    void testInvariant() const; 
    template<bool SKIP_INITIALIZATION, class ShapeIterator>
        void resizeHelper(ShapeIterator, ShapeIterator, const T& = T());

    allocator_type dataAllocator_;
};

// implementation of View

#ifdef HAVE_CPP11_INITIALIZER_LISTS
/// Compute the index that corresponds to a sequence of coordinates.
///
/// \param coordinate Coordinate given as initializer list.
/// \param out Index (output)
/// \sa coordinatesToOffset(), indexToCoordinates(), and indexToOffset()
///
template<class T, bool isConst, class A>
inline void
View<T, isConst, A>::coordinatesToIndex
(
    std::initializer_list<std::size_t> coordinate,
    std::size_t& out
) const 
{
    coordinatesToIndex(coordinate.begin(), out);
}
#endif

/// Compute the index that corresponds to a sequence of coordinates.
///
/// \param it An iterator to the beginning of the coordinate sequence.
/// \param out Index (output)
/// \sa coordinatesToOffset(), indexToCoordinates(), and indexToOffset()
///
template<class T, bool isConst, class A> 
template<class CoordinateIterator>
inline void 
View<T, isConst, A>::coordinatesToIndex
(
    CoordinateIterator it,
    std::size_t& out
) const
{
    testInvariant();
    out = 0;
    for(std::size_t j=0; j<this->dimension(); ++j, ++it) {
        marray_detail::Assert(MARRAY_NO_ARG_TEST || static_cast<std::size_t>(*it) < shape(j));
        out += static_cast<std::size_t>(*it) * geometry_.shapeStrides(j);
    }
}

#ifdef HAVE_CPP11_INITIALIZER_LISTS
/// Compute the offset that corresponds to a sequence of coordinates.
///
/// \param it An iterator to the beginning of the coordinate sequence.
/// \param out Index (output)
/// \sa coordinatesToIndex(), indexToCoordinates(), and indexToOffset()
///
template<class T, bool isConst, class A> 
inline void
View<T, isConst, A>::coordinatesToOffset
(
    std::initializer_list<std::size_t> coordinate,
    std::size_t& out
) const
{
    coordinatesToOffset(coordinate.begin(), out);
}
#endif

/// Compute the offset that corresponds to a sequence of coordinates.
///
/// \param it An iterator to the beginning of the coordinate sequence.
/// \param out Index (output)
/// \sa coordinatesToIndex(), indexToCoordinates(), and indexToOffset()
///
template<class T, bool isConst, class A> 
template<class CoordinateIterator>
inline void
View<T, isConst, A>::coordinatesToOffset
(
    CoordinateIterator it,
    std::size_t& out
) const
{
    testInvariant();
    out = 0;
    for(std::size_t j=0; j<this->dimension(); ++j, ++it) {
        marray_detail::Assert(MARRAY_NO_ARG_TEST || static_cast<std::size_t>(*it) < shape(j));
        out += static_cast<std::size_t>(*it) * strides(j);
    }
}

/// Compute the coordinate sequence that corresponds to an index.
///
/// \param index Index
/// \param outit An iterator into a container into which the coordinate
/// sequence is to be written (output).
/// \sa coordinatesToIndex(), coordinatesToOffset(), and indexToOffset()
///
template<class T, bool isConst, class A>
template<class CoordinateIterator>
inline void
View<T, isConst, A>::indexToCoordinates
(
    std::size_t index, // copy to work on
    CoordinateIterator outit
) const
{
    testInvariant();
    marray_detail::Assert(MARRAY_NO_DEBUG || this->dimension() > 0);
    marray_detail::Assert(MARRAY_NO_ARG_TEST || index < this->size());
    if(coordinateOrder() == FirstMajorOrder) {
        for(std::size_t j=0; j<this->dimension(); ++j, ++outit) {
            *outit = std::size_t(index / geometry_.shapeStrides(j));
            index = index % geometry_.shapeStrides(j);
        }
    }
    else { // last major order
        std::size_t j = this->dimension()-1;
        outit += j;
        for(;;) {
            *outit = std::size_t(index / geometry_.shapeStrides(j));
            index = index % geometry_.shapeStrides(j);
            if(j == 0) {
                break;
            }
            else {
                --outit;
                --j;
            }
        }
    }
}

/// Compute the offset that corresponds to an index.
///
/// \param index Index.
/// \param out Offset (output).
/// \sa coordinatesToIndex(), coordinatesToOffset(), and indexToCoordinates()
///
template<class T, bool isConst, class A> 
inline void
View<T, isConst, A>::indexToOffset
(
    std::size_t index,
    std::size_t& out
) const
{
    testInvariant();
    marray_detail::Assert(MARRAY_NO_ARG_TEST || index < this->size()); 
    if(isSimple()) {
        out = index;
    }
    else {
        out = 0;
        if(coordinateOrder() == FirstMajorOrder) {
            for(std::size_t j=0; j<this->dimension(); ++j) {
                out += geometry_.strides(j) * (index / geometry_.shapeStrides(j));
                index = index % geometry_.shapeStrides(j);
            }
        }
        else { // last major order
            if(this->dimension() == 0) {
                marray_detail::Assert(MARRAY_NO_ARG_TEST || index == 0);
                return;
            }
            else {
                std::size_t j = this->dimension()-1;
                for(;;) {
                    out += geometry_.strides(j) * (index / geometry_.shapeStrides(j));
                    index = index % geometry_.shapeStrides(j);
                    if(j == 0) {
                        break;
                    }
                    else {
                        --j;
                    }
                }
            }
        }
    }
}

/// Empty constructor.
///
/// The empty constructor sets the data pointer to 0.
/// It does not allocate memory for a scalar.
///
/// \param allocator Allocator.
///
template<class T, bool isConst, class A> 
inline
View<T, isConst, A>::View
(
    const allocator_type& allocator
)
: data_(0),
  geometry_(geometry_type(allocator))
{
    testInvariant();
}

// private constructor
template<class T, bool isConst, class A> 
inline
View<T, isConst, A>::View
(
    pointer data,
    const geometry_type& geometry
)
: data_(data), 
  geometry_(geometry)
{
    testInvariant();
}

/// Construct View from a scalar.
///
/// \param data Pointer to data.
/// \param allocator Allocator.
///
template<class T, bool isConst, class A> 
inline
View<T, isConst, A>::View
(
    pointer data,
    const allocator_type& allocator
)
: data_(data),
  geometry_(geometry_type(0, defaultOrder, 1, true, allocator))
{
    testInvariant();
}

/// Construct View from a View on mutable data.
///
/// \param in View on mutable data.
///
template<class T, bool isConst, class A> 
inline
View<T, isConst, A>::View
(
    const View<T, false, A>& in
)
: data_(in.data_),
  geometry_(in.geometry_)
{
    testInvariant();
}

/// Construct unstrided View
/// 
/// \param begin Iterator to the beginning of a sequence that
/// defines the shape.
/// \param end Iterator to the end of this sequence.
/// \param data Pointer to data.
/// \param externalCoordinateOrder Flag specifying the order
/// of coordinates based on which the strides are computed.
/// \param internalCoordinateOrder Flag specifying the order
/// of coordinates used for scalar indexing and iterators.
/// \param allocator Allocator.
///

template<class T, bool isConst, class A> 
template<class ShapeIterator>
inline
View<T, isConst, A>::View
(
    ShapeIterator begin,
    ShapeIterator end,
    pointer data,
    const CoordinateOrder& externalCoordinateOrder,
    const CoordinateOrder& internalCoordinateOrder,
    const allocator_type& allocator
) 
:   data_(data),
    geometry_(begin, end, externalCoordinateOrder, 
        internalCoordinateOrder, allocator)
{
    testInvariant();
}

/// Construct strided View
/// 
/// \param begin Iterator to the beginning of a sequence that
/// defines the shape.
/// \param end Iterator to the end of this sequence.
/// \param it Iterator to the beginning of a sequence that
/// defines the strides.
/// \param data Pointer to data.
/// \param internalCoordinateOrder Flag specifying the order
/// of coordinates used for scalar indexing and iterators.
/// \param allocator Allocator.
///
template<class T, bool isConst, class A> 
template<class ShapeIterator, class StrideIterator>
inline
View<T, isConst, A>::View
(
    ShapeIterator begin,
    ShapeIterator end,
    StrideIterator it,
    pointer data,
    const CoordinateOrder& internalCoordinateOrder,
    const allocator_type& allocator
)
: data_(data),
  geometry_(begin, end, it, internalCoordinateOrder, allocator)  
{
    testInvariant();
}

#ifdef HAVE_CPP11_INITIALIZER_LISTS
/// Construct unstrided View
/// 
/// \param shape Shape initializer list.
/// \param data Pointer to data.
/// \param externalCoordinateOrder Flag specifying the order
/// of coordinates based on which the strides are computed.
/// \param internalCoordinateOrder Flag specifying the order
/// of coordinates used for scalar indexing and iterators.
/// \param allocator Allocator.
///
template<class T, bool isConst, class A>
inline
View<T, isConst, A>::View
(
    std::initializer_list<std::size_t> shape,
    pointer data,
    const CoordinateOrder& externalCoordinateOrder,
    const CoordinateOrder& internalCoordinateOrder,
    const allocator_type& allocator
)
:   data_(data),
    geometry_(shape.begin(), shape.end(), externalCoordinateOrder, 
              internalCoordinateOrder, allocator)  
{
    testInvariant();
}

/// Construct strided View
/// 
/// \param shape Shape initializer list.
/// \param strides Strides initializer list.
/// \param data Pointer to data.
/// \param internalCoordinateOrder Flag specifying the order
/// of coordinates used for scalar indexing and iterators.
///
template<class T, bool isConst, class A> 
inline
View<T, isConst, A>::View
(
    std::initializer_list<std::size_t> shape,
    std::initializer_list<std::size_t> strides,
    pointer data,
    const CoordinateOrder& internalCoordinateOrder,
    const allocator_type& allocator
)
:   data_(data),
    geometry_(shape.begin(), shape.end(), strides.begin(), 
              internalCoordinateOrder, allocator)
{
    testInvariant();
}
#endif

/// Clear View.
///
/// Leaves the View in the same state as if the empty constructor
/// had been called.
///
/// \sa View()
///
template<class T, bool isConst, class A> 
inline void
View<T, isConst, A>::assign
(
    const allocator_type& allocator
)
{
    geometry_ = geometry_type(allocator);
    data_ = 0;
    testInvariant();
}

/// Initialize unstrided View
/// 
/// \param begin Iterator to the beginning of a sequence that
/// defines the shape.
/// \param end Iterator to the end of this sequence.
/// \param data Pointer to data.
/// \param externalCoordinateOrder Flag specifying the order
/// of coordinates based on which the strides are computed.
/// \param internalCoordinateOrder Flag specifying the order
/// of coordinates used for scalar indexing and iterators.
/// \param allocator Allocator.
///
template<class T, bool isConst, class A> 
template<class ShapeIterator>
inline void
View<T, isConst, A>::assign
(
    ShapeIterator begin,
    ShapeIterator end,
    pointer data,
    const CoordinateOrder& externalCoordinateOrder,
    const CoordinateOrder& internalCoordinateOrder,
    const allocator_type& allocator
)
{
    // the invariant is not tested as a pre-condition of this
    // function to allow for unsafe manipulations prior to its
    // call
    geometry_ = typename marray_detail::Geometry<A>(begin, end, 
        externalCoordinateOrder, internalCoordinateOrder, allocator);
    data_ = data;
    testInvariant();    
}

/// Initialize strided View
/// 
/// \param begin Iterator to the beginning of a sequence that
/// defines the shape.
/// \param end Iterator to the end of this sequence.
/// \param it Iterator to the beginning of a sequence that
/// defines the strides.
/// \param data Pointer to data.
/// \param internalCoordinateOrder Flag specifying the order
/// of coordinates used for scalar indexing and iterators.
/// \param allocator Allocator.
///
template<class T, bool isConst, class A> 
template<class ShapeIterator, class StrideIterator>
inline void
View<T, isConst, A>::assign
(
    ShapeIterator begin,
    ShapeIterator end,
    StrideIterator it,
    pointer data,
    const CoordinateOrder& internalCoordinateOrder,
    const allocator_type& allocator
)
{
    // the invariant is not tested as a pre-condition of this
    // function to allow for unsafe manipulations prior to its
    // call
    geometry_ = typename marray_detail::Geometry<A>(begin, end, 
        it, internalCoordinateOrder, allocator);
    data_ = data;
    testInvariant();
}

#ifdef HAVE_CPP11_INITIALIZER_LISTS
/// Initialize unstrided View
/// 
/// \param shape Shape initializer list.
/// \param data Pointer to data.
/// \param externalCoordinateOrder Flag specifying the order
/// of coordinates based on which the strides are computed.
/// \param internalCoordinateOrder Flag specifying the order
/// of coordinates used for scalar indexing and iterators.
/// \param allocator Allocator.
///
template<class T, bool isConst, class A> 
inline void
View<T, isConst, A>::assign
(
    std::initializer_list<std::size_t> shape,
    pointer data,
    const CoordinateOrder& externalCoordinateOrder,
    const CoordinateOrder& internalCoordinateOrder,
    const allocator_type& allocator
)
{
    assign(shape.begin(), shape.end(), data, externalCoordinateOrder,
        internalCoordinateOrder, allocator);
}

/// Initialize strided View
/// 
/// \param shape Shape initializer list.
/// \param strides Strides initialier list.
/// defines the strides.
/// \param data Pointer to data.
/// \param internalCoordinateOrder Flag specifying the order
/// of coordinates used for scalar indexing and iterators.
/// \param allocator Allocator.
///
template<class T, bool isConst, class A> 
inline void
View<T, isConst, A>::assign
(
    std::initializer_list<std::size_t> shape,
    std::initializer_list<std::size_t> strides,
    pointer data,
    const CoordinateOrder& internalCoordinateOrder,
    const allocator_type& allocator
)
{
    assign(shape.begin(), shape.end(), strides.begin(), data,
        internalCoordinateOrder, allocator);
}
#endif

/// Reference data.
///
/// \param u If u is an integer type, scalar indexing is performed.
/// Otherwise, it is assumed that u is an iterator to the beginning
/// of a coordinate sequence. 
/// \return Reference to the entry at u.
///
template<class T, bool isConst, class A> 
template<class U>
inline typename View<T, isConst, A>::reference
View<T, isConst, A>::operator()
(
    U u
)
{
    return marray_detail::AccessOperatorHelper<std::numeric_limits<U>::is_integer>::execute(*this, u);
}

/// Reference data.
///
/// \param u If u is an integer type, scalar indexing is performed.
/// Otherwise, it is assumed that u is an iterator to the beginning
/// of a coordinate sequence. 
/// \return Reference to the entry at u.
///
template<class T, bool isConst, class A> 
template<class U>
inline typename View<T, isConst, A>::reference
View<T, isConst, A>::operator()
(
    U u
) const
{
    return marray_detail::AccessOperatorHelper<std::numeric_limits<U>::is_integer>::execute(*this, u);
}

#ifndef HAVE_CPP11_VARIADIC_TEMPLATES

/// Reference data in a 2-dimensional View by coordinates.
///
/// This function issues a runtime error if the View is not
/// 2-dimensional.
///
/// \param c0 1st coordinate. 
/// \param c1 2nd coordinate.
///
template<class T, bool isConst, class A> 
inline typename View<T, isConst, A>::reference
View<T, isConst, A>::operator()
(
    const std::size_t c0,
    const std::size_t c1
)
{
    testInvariant();
    marray_detail::Assert(MARRAY_NO_DEBUG || (data_ != 0 && dimension() == 2));
    marray_detail::Assert(MARRAY_NO_ARG_TEST || (c0 < shape(0) && c1 < shape(1)));
    return data_[c0*strides(0) + c1*strides(1)];
}

/// Reference data in a 2-dimensional View by coordinates.
///
/// This function issues a runtime error if the View is not
/// 2-dimensional.
///
/// \param c0 1st coordinate. 
/// \param c1 2nd coordinate.
///
template<class T, bool isConst, class A> 
inline typename View<T, isConst, A>::reference
View<T, isConst, A>::operator()
(
    const std::size_t c0,
    const std::size_t c1
) const
{
    testInvariant();
    marray_detail::Assert(MARRAY_NO_DEBUG || (data_ != 0 && dimension() == 2));
    marray_detail::Assert(MARRAY_NO_ARG_TEST || (c0 < shape(0) && c1 < shape(1)));
    return data_[c0*strides(0) + c1*strides(1)];
}

/// Reference data in a 3-dimensional View by coordinates.
///
/// This function issues a runtime error if the View is not
/// 3-dimensional.
///
/// \param c0 1st coordinate. 
/// \param c1 2nd coordinate.
/// \param c2 3rd coordinate.
///
template<class T, bool isConst, class A> 
inline typename View<T, isConst, A>::reference
View<T, isConst, A>::operator()
(
    const std::size_t c0,
    const std::size_t c1,
    const std::size_t c2
)
{
    testInvariant();
    marray_detail::Assert(MARRAY_NO_DEBUG || (data_ != 0 && dimension() == 3));
    marray_detail::Assert(MARRAY_NO_ARG_TEST || (c0 < shape(0) && c1 < shape(1)
        && c2 < shape(2)));
    return data_[c0*strides(0) + c1*strides(1) + c2*strides(2)];
}

/// Reference data in a 3-dimensional View by coordinates.
///
/// This function issues a runtime error if the View is not
/// 3-dimensional.
///
/// \param c0 1st coordinate. 
/// \param c1 2nd coordinate.
/// \param c2 3rd coordinate.
///
template<class T, bool isConst, class A> 
inline typename View<T, isConst, A>::reference
View<T, isConst, A>::operator()
(
    const std::size_t c0,
    const std::size_t c1,
    const std::size_t c2
) const
{
    testInvariant();
    marray_detail::Assert(MARRAY_NO_DEBUG || (data_ != 0 && dimension() == 3));
    marray_detail::Assert(MARRAY_NO_ARG_TEST || (c0 < shape(0) && c1 < shape(1)
        && c2 < shape(2)));
    return data_[c0*strides(0) + c1*strides(1) + c2*strides(2)];
}

/// Reference data in a 4-dimensional View by coordinates.
///
/// This function issues a runtime error if the View is not
/// 4-dimensional.
///
/// \param c0 1st coordinate. 
/// \param c1 2nd coordinate.
/// \param c2 3rd coordinate.
/// \param c3 4th coordinate.
///
template<class T, bool isConst, class A> 
inline typename View<T, isConst, A>::reference
View<T, isConst, A>::operator()
(
    const std::size_t c0,
    const std::size_t c1,
    const std::size_t c2,
    const std::size_t c3
)
{
    testInvariant();
    marray_detail::Assert(MARRAY_NO_DEBUG || (data_ != 0 && dimension() == 4));
    marray_detail::Assert(MARRAY_NO_ARG_TEST || (c0 < shape(0) && c1 < shape(1)
        && c2 < shape(2) && c3 < shape(3)));
    return data_[c0*strides(0) + c1*strides(1) + c2*strides(2) 
        + c3*strides(3)];
}

/// Reference data in a 4-dimensional View by coordinates.
///
/// This function issues a runtime error if the View is not
/// 4-dimensional.
///
/// \param c0 1st coordinate. 
/// \param c1 2nd coordinate.
/// \param c2 3rd coordinate.
/// \param c3 4th coordinate.
///
template<class T, bool isConst, class A> 
inline typename View<T, isConst, A>::reference
View<T, isConst, A>::operator()
(
    const std::size_t c0,
    const std::size_t c1,
    const std::size_t c2,
    const std::size_t c3
) const
{
    testInvariant();
    marray_detail::Assert(MARRAY_NO_DEBUG || (data_ != 0 && dimension() == 4));
    marray_detail::Assert(MARRAY_NO_ARG_TEST || (c0 < shape(0) && c1 < shape(1)
        && c2 < shape(2) && c3 < shape(3)));
    return data_[c0*strides(0) + c1*strides(1) + c2*strides(2) 
        + c3*strides(3)];
}

/// Reference data in a 5-dimensional View by coordinates.
///
/// This function issues a runtime error if the View is not
/// 5-dimensional.
///
/// \param c0 1st coordinate. 
/// \param c1 2nd coordinate.
/// \param c2 3rd coordinate.
/// \param c3 4th coordinate.
/// \param c4 5th coordinate.
///
template<class T, bool isConst, class A> 
inline typename View<T, isConst, A>::reference
View<T, isConst, A>::operator()
(
    const std::size_t c0,
    const std::size_t c1,
    const std::size_t c2,
    const std::size_t c3,
    const std::size_t c4
)
{
    testInvariant();
    marray_detail::Assert(MARRAY_NO_DEBUG || (data_ != 0 && dimension() == 5));
    marray_detail::Assert(MARRAY_NO_ARG_TEST || (c0 < shape(0) && c1 < shape(1)
        && c2 < shape(2) && c3 < shape(3) && c4 < shape(4)));
    return data_[c0*strides(0) + c1*strides(1) + c2*strides(2) 
        + c3*strides(3) + c4*strides(4)];
}

/// Reference data in a 5-dimensional View by coordinates.
///
/// This function issues a runtime error if the View is not
/// 5-dimensional.
///
/// \param c0 1st coordinate. 
/// \param c1 2nd coordinate.
/// \param c2 3rd coordinate.
/// \param c3 4th coordinate.
/// \param c4 5th coordinate.
///
template<class T, bool isConst, class A> 
inline typename View<T, isConst, A>::reference
View<T, isConst, A>::operator()
(
    const std::size_t c0,
    const std::size_t c1,
    const std::size_t c2,
    const std::size_t c3,
    const std::size_t c4
) const
{
    testInvariant();
    marray_detail::Assert(MARRAY_NO_DEBUG || (data_ != 0 && dimension() == 5));
    marray_detail::Assert(MARRAY_NO_ARG_TEST || (c0 < shape(0) && c1 < shape(1)
        && c2 < shape(2) && c3 < shape(3) && c4 < shape(4)));
    return data_[c0*strides(0) + c1*strides(1) + c2*strides(2) 
        + c3*strides(3) + c4*strides(4)];
}

/// Reference data in a 10-dimensional View by coordinates.
///
/// This function issues a runtime error if the View is not

/// 5-dimensional.
///
/// \param c0 1st coordinate. 
/// \param c1 2nd coordinate.
/// \param c2 3rd coordinate.
/// \param c3 4th coordinate.
/// \param c4 5th coordinate.
/// \param c5 6th coordinate.
/// \param c6 7th coordinate.
/// \param c7 8th coordinate.
/// \param c8 9th coordinate.
/// \param c9 10th coordinate.
///
template<class T, bool isConst, class A> 
inline typename View<T, isConst, A>::reference
View<T, isConst, A>::operator()
(
    const std::size_t c0,
    const std::size_t c1,
    const std::size_t c2,
    const std::size_t c3,
    const std::size_t c4,
    const std::size_t c5,
    const std::size_t c6,
    const std::size_t c7,
    const std::size_t c8,
    const std::size_t c9
) 
{
    testInvariant();
    marray_detail::Assert(MARRAY_NO_DEBUG || (data_ != 0 && dimension() == 10));
    marray_detail::Assert(MARRAY_NO_ARG_TEST || (c0 < shape(0) && c1 < shape(1)
        && c2 < shape(2) && c3 < shape(3) && c4 < shape(4)
        && c5 < shape(5) && c6 < shape(6) && c7 < shape(7)
        && c8 < shape(8) && c9 < shape(9)));
    return data_[c0*strides(0) + c1*strides(1) + c2*strides(2)
        + c3*strides(3) + c4*strides(4) + c5*strides(5) + c6*strides(6) 
        + c7*strides(7) + c8*strides(8) + c9*strides(9)];
}

/// Reference data in a 10-dimensional View by coordinates.
///
/// This function issues a runtime error if the View is not
/// 5-dimensional.
///
/// \param c0 1st coordinate. 
/// \param c1 2nd coordinate.
/// \param c2 3rd coordinate.
/// \param c3 4th coordinate.
/// \param c4 5th coordinate.
/// \param c5 6th coordinate.
/// \param c6 7th coordinate.
/// \param c7 8th coordinate.
/// \param c8 9th coordinate.
/// \param c9 10th coordinate.
///
template<class T, bool isConst, class A> 
inline typename View<T, isConst, A>::reference
View<T, isConst, A>::operator()
(
    const std::size_t c0,
    const std::size_t c1,
    const std::size_t c2,
    const std::size_t c3,
    const std::size_t c4,
    const std::size_t c5,
    const std::size_t c6,
    const std::size_t c7,
    const std::size_t c8,
    const std::size_t c9
) const
{
    testInvariant();
    marray_detail::Assert(MARRAY_NO_DEBUG || (data_ != 0 && dimension() == 10));
    marray_detail::Assert(MARRAY_NO_ARG_TEST || (c0 < shape(0) && c1 < shape(1)
        && c2 < shape(2) && c3 < shape(3) && c4 < shape(4)
        && c5 < shape(5) && c6 < shape(6) && c7 < shape(7)
        && c8 < shape(8) && c9 < shape(9)));
    return data_[c0*strides(0) + c1*strides(1) + c2*strides(2)
        + c3*strides(3) + c4*strides(4) + c5*strides(5) + c6*strides(6) 
        + c7*strides(7) + c8*strides(8) + c9*strides(9)];
}

#else

template<class T, bool isConst, class A>
inline std::size_t
View<T, isConst, A>::elementAccessHelper
(
    const std::size_t Dim, 
    const std::size_t value    
)
{
    marray_detail::Assert(MARRAY_NO_ARG_TEST || (value < shape(Dim-1) ) );
    return strides(Dim-1) * value;
}

template<class T, bool isConst, class A>
inline std::size_t
View<T, isConst, A>::elementAccessHelper
(
    const std::size_t Dim, 
    const std::size_t value
) const
{
    marray_detail::Assert(MARRAY_NO_ARG_TEST || (value < shape(Dim-1) ) );  
    return strides(Dim-1) * value;
}

template<class T, bool isConst, class A>
template<typename... Args>
inline std::size_t
View<T, isConst, A>::elementAccessHelper
(
    const std::size_t Dim, 
    const std::size_t value, 
    const Args... args
)
{
    marray_detail::Assert(MARRAY_NO_ARG_TEST || (value < shape(Dim-1-sizeof...(args)) ) );      
    return value * strides(Dim-1-sizeof...(args)) + elementAccessHelper(Dim, args...); 
}

template<class T, bool isConst, class A>
template<typename... Args>
inline std::size_t
View<T, isConst, A>::elementAccessHelper
(
    const std::size_t Dim, 
    const std::size_t value, 
    const Args... args
) const
{
    marray_detail::Assert(MARRAY_NO_ARG_TEST || (value < shape(Dim-1-sizeof...(args)) ) );  
    return value * strides(Dim-1-sizeof...(args)) + elementAccessHelper(Dim, args...); 
}

template<class T, bool isConst, class A>
inline typename View<T, isConst, A>::reference
View<T, isConst, A>::operator()
(
    const std::size_t value
)
{
    testInvariant();
    if(dimension() == 0) {
        marray_detail::Assert(MARRAY_NO_ARG_TEST || value == 0);
        return data_[0];
    }
    else {
        std::size_t offset;
        indexToOffset(value, offset);
        return data_[offset];
    }
}

template<class T, bool isConst, class A>
inline typename View<T, isConst, A>::reference
View<T, isConst, A>::operator()
(
    const std::size_t value
) const
{
    testInvariant();    
    if(dimension() == 0) {
        marray_detail::Assert(MARRAY_NO_ARG_TEST || value == 0);
        return data_[0];
    }
    else {
        std::size_t offset;
        indexToOffset(value, offset);
        return data_[offset];
    }
}

template<class T, bool isConst, class A>
template<typename... Args>
inline typename View<T, isConst, A>::reference
View<T, isConst, A>::operator()
(
    const std::size_t value, 
    const Args... args
)
{
    testInvariant();
    marray_detail::Assert( MARRAY_NO_DEBUG || ( data_ != 0 && sizeof...(args)+1 == dimension() ) );
    return data_[strides(0)*value + elementAccessHelper(sizeof...(args)+1, args...) ];
}

template<class T, bool isConst, class A>
template<typename... Args>
inline typename View<T, isConst, A>::reference
View<T, isConst, A>::operator()
(
    const std::size_t value, 
    const Args... args
) const
{
    testInvariant();
    marray_detail::Assert( MARRAY_NO_DEBUG || ( data_ != 0 && sizeof...(args)+1 == dimension() ) );
    return data_[ strides(0) * static_cast<std::size_t>(value) 
        + static_cast<std::size_t>(elementAccessHelper(sizeof...(args)+1, args...)) ];
}

#endif // #ifndef HAVE_CPP11_VARIADIC_TEMPLATES

/// Get the number of data items.
///
/// \return Size.
///
template<class T, bool isConst, class A> 
inline const std::size_t
View<T, isConst, A>::size() const
{
    return geometry_.size();
}

/// Get the dimension.
///
/// Not well-defined if the data pointer is 0.
///
/// \return Dimension.
///
template<class T, bool isConst, class A> 
inline const std::size_t
View<T, isConst, A>::dimension() const
{
    marray_detail::Assert(MARRAY_NO_DEBUG || this->data_ != 0);
    return geometry_.dimension();
}

/// Get the shape in one dimension.
///
/// \param dimension Dimension
/// \return Shape in that dimension.
///
template<class T, bool isConst, class A> 
inline const std::size_t
View<T, isConst, A>::shape
(
    const std::size_t dimension
) const
{
    testInvariant();
    marray_detail::Assert(MARRAY_NO_DEBUG || data_ != 0);
    marray_detail::Assert(MARRAY_NO_ARG_TEST || dimension < this->dimension());
    return geometry_.shape(dimension);
}

/// Get a constant iterator to the beginning of the shape vector.
///
/// \return iterator.
/// \sa shapeEnd()
///
template<class T, bool isConst, class A> 
inline const std::size_t*
View<T, isConst, A>::shapeBegin() const
{
    testInvariant();
    marray_detail::Assert(MARRAY_NO_DEBUG || data_ != 0);
    return geometry_.shapeBegin();
}

/// Get a constant iterator to the end of the shape vector.
///
/// \return iterator.
/// \sa shapeBegin()
///
template<class T, bool isConst, class A> 
inline const std::size_t*
View<T, isConst, A>::shapeEnd() const
{
    testInvariant();
    marray_detail::Assert(MARRAY_NO_DEBUG || data_ != 0);
    return geometry_.shapeEnd();
}

/// Get the strides in one dimension.
///
/// \param dimension Dimension
/// \return Stride in that dimension.
///
template<class T, bool isConst, class A> 
inline const std::size_t
View<T, isConst, A>::strides
(
    const std::size_t dimension
) const
{
    testInvariant();
    marray_detail::Assert(MARRAY_NO_DEBUG || data_ != 0);
    marray_detail::Assert(MARRAY_NO_ARG_TEST || dimension < this->dimension());
    return geometry_.strides(dimension);
}

/// Get a constant iterator to the beginning of the strides vector.
///
/// \return iterator.
/// \sa stridesEnd()
///
template<class T, bool isConst, class A> 
inline const std::size_t*
View<T, isConst, A>::stridesBegin() const
{
    testInvariant();
    marray_detail::Assert(MARRAY_NO_DEBUG || data_ != 0);
    return geometry_.stridesBegin();
}

/// Get a constant iterator to the end of the strides vector.
///
/// \return iterator.
/// \sa stridesBegin()
///
template<class T, bool isConst, class A> 
inline const std::size_t*
View<T, isConst, A>::stridesEnd() const
{
    testInvariant();
    marray_detail::Assert(MARRAY_NO_DEBUG || data_ != 0);
    return geometry_.stridesEnd();
}

/// Get the coordinate order used for scalar indexing and iterators.
///
/// \return CoordinateOrder. enum: FirstMajorOrder, LastMajorOrder
///
template<class T, bool isConst, class A> 
inline const CoordinateOrder&
View<T, isConst, A>::coordinateOrder() const
{
    testInvariant();
    return geometry_.coordinateOrder();
}

/// Determine whether the shape strides equal the strides of the View.
///
/// \return bool.
///
template<class T, bool isConst, class A> 
inline const bool
View<T, isConst, A>::isSimple() const
{
    testInvariant();
    return geometry_.isSimple();
}

/// Assignment.
/// 
/// operator= (the assignment operator) has a non-trivial behavior.
/// In most cases, it will work as most programmers will expect.
/// Here's a complete description of the semantics of to.operator=(from)
/// or equivalently, to = from.
/// 
/// Consider the following cases:
/// (1) 'to' is mutable (isConst == false)
///     (a) 'from' is mutable (isConst == false)
///         (i) 'to' is initialized (data_ != 0)
///         (ii) 'to' is un-initialized (data_ == 0)
///     (b) 'from' is constant (isConst == true)
/// (2) 'to' is constant (isConst == true)
/// 
/// (i) The operator attempts to copy the data under view 'b' to
/// the memory under view 'a'. This works if both views have the
/// same size, regardless of their dimension and shape. Equality
/// of sizes is checked by an assertion.
/// 
/// (ii) Unless &a == &b (self-assignment), the operator copies
/// the (data) pointer of view 'b' to view 'a', without copying
/// the data itself. In addition, all the properties of view 'b'
/// are copied to view 'a'.
/// 
/// (b) The operator attempts to copy the data under view 'b' to
/// the memory under view 'a'. This works if both views have the
/// same size, regardless of their dimension and shape. Equality
/// of sizes is checked by an assertion. If 'a' is un-initialized
/// the assertion fails (because the size of a will be zero).
/// Unlike in (ii), the pointer is not copied in this case.
/// Thus, a conversion from mutable to const is prevented.
/// 
/// (2) Unless &a == &b (self-assignment), the operator copies
/// the (data) pointer of view 'b' to view 'a', without copying
/// the data itself. In addition, all the properties of view 'b'
/// are copied to view 'a'. Note that changing the data under
/// a constant view would be counter-intuitive.
/// 
template<class T, bool isConst, class A> 
inline View<T, isConst, A>&
View<T, isConst, A>::operator=
(
    const View<T, true, A>& in
)
{
    testInvariant();
    marray_detail::AssignmentOperatorHelper<isConst, T, T, A, A>::execute(in, *this);
    testInvariant();
    return *this;
}

/// Assignment.
///
template<class T, bool isConst, class A> 
inline View<T, isConst, A>&
View<T, isConst, A>::operator=
(
    const View<T, false, A>& in
)
{
    testInvariant();
    marray_detail::AssignmentOperatorHelper<isConst, T, T, A, A>::execute(in, *this);
    testInvariant();
    return *this;
}

/// Assignment.
///
template<class T, bool isConst, class A> 
template<class TLocal, bool isConstLocal, class ALocal> 
inline View<T, isConst, A>&
View<T, isConst, A>::operator=
(
    const View<TLocal, isConstLocal, ALocal>& in
)
{
    testInvariant();
    marray_detail::AssignmentOperatorHelper<isConst, TLocal, T, ALocal, A>::execute(in, *this);
    testInvariant();
    return *this;
}

/// Assignment.
///
/// \param value Value.
///
/// All entries are set to value.
///
template<class T, bool isConst, class A> 
inline View<T, isConst, A>& 
View<T, isConst, A>::operator=
(
    const T& value
)
{
    marray_detail::Assert(MARRAY_NO_DEBUG || data_ != 0);
    if(isSimple()) {
        for(std::size_t j=0; j<geometry_.size(); ++j) {
            data_[j] = value;
        }
    }
    else if(dimension() == 1)
        marray_detail::OperateHelperBinaryScalar<1, marray_detail::Assign<T, T>, T, T, A>::operate(*this, value, marray_detail::Assign<T, T>(), data_);
    else if(dimension() == 2)
        marray_detail::OperateHelperBinaryScalar<2, marray_detail::Assign<T, T>, T, T, A>::operate(*this, value, marray_detail::Assign<T, T>(), data_);
    else if(dimension() == 3)
        marray_detail::OperateHelperBinaryScalar<3, marray_detail::Assign<T, T>, T, T, A>::operate(*this, value, marray_detail::Assign<T, T>(), data_);
    else if(dimension() == 4)
        marray_detail::OperateHelperBinaryScalar<4, marray_detail::Assign<T, T>, T, T, A>::operate(*this, value, marray_detail::Assign<T, T>(), data_);
    else if(dimension() == 5)
        marray_detail::OperateHelperBinaryScalar<5, marray_detail::Assign<T, T>, T, T, A>::operate(*this, value, marray_detail::Assign<T, T>(), data_);
    else if(dimension() == 6)
        marray_detail::OperateHelperBinaryScalar<6, marray_detail::Assign<T, T>, T, T, A>::operate(*this, value, marray_detail::Assign<T, T>(), data_);
    else if(dimension() == 7)
        marray_detail::OperateHelperBinaryScalar<7, marray_detail::Assign<T, T>, T, T, A>::operate(*this, value, marray_detail::Assign<T, T>(), data_);
    else if(dimension() == 8)
        marray_detail::OperateHelperBinaryScalar<8, marray_detail::Assign<T, T>, T, T, A>::operate(*this, value, marray_detail::Assign<T, T>(), data_);
    else if(dimension() == 9)
        marray_detail::OperateHelperBinaryScalar<9, marray_detail::Assign<T, T>, T, T, A>::operate(*this, value, marray_detail::Assign<T, T>(), data_);
    else if(dimension() == 10)
        marray_detail::OperateHelperBinaryScalar<10, marray_detail::Assign<T, T>, T, T, A>::operate(*this, value, marray_detail::Assign<T, T>(), data_);
    else {
        for(iterator it = begin(); it.hasMore(); ++it) {
            *it = value;
        }
    }
    return *this;
}

template<class T, bool isConst, class A> 
template<class E, class Te>
inline View<T, isConst, A>& 
View<T, isConst, A>::operator=
(
    const ViewExpression<E, Te>& expression
)
{
    marray_detail::operate(*this, expression, marray_detail::Assign<T, Te>());
    return *this;
}

/// Get a sub-view with the same coordinate order.
///
/// \param bit Iterator to the beginning of a coordinate sequence
/// that determines the start position of the sub-view.
/// \param sit Iterator to the beginning of a sequence
/// that determines the shape of the sub-view.
/// \param out Sub-View (output).
///
template<class T, bool isConst, class A> 
template<class BaseIterator, class ShapeIterator>
inline void
View<T, isConst, A>::view
(
    BaseIterator bit,
    ShapeIterator sit,
    View<T, isConst, A>& out
) const
{
    view(bit, sit, coordinateOrder(), out);
}

/// Get a sub-view.
///
/// \param bit Iterator to the beginning of a coordinate sequence
/// that determines the start position of the sub-view.
/// \param sit Iterator to the beginning of a sequence
/// that determines the shape of the sub-view.
/// \param internalCoordinateOrder Flag to set the coordinate order
/// for scalar indexing and iterators of the sub-view.
/// \param out Sub-View (output).
///
template<class T, bool isConst, class A> 
template<class BaseIterator, class ShapeIterator>
inline void
View<T, isConst, A>::view
(
    BaseIterator bit,
    ShapeIterator sit,
    const CoordinateOrder& internalCoordinateOrder,
    View<T, isConst, A>& out
) const
{
    testInvariant();
    std::size_t offset = 0;
    coordinatesToOffset(bit, offset);
    out.assign(sit, sit+dimension(), geometry_.stridesBegin(),
        data_+offset, internalCoordinateOrder);
}

/// Get a sub-view with the same coordinate order.
///
/// \param bit Iterator to the beginning of a coordinate sequence
/// that determines the start position of the sub-view.
/// \param sit Iterator to the beginning of a sequence
/// that determines the shape of the sub-view.
/// \return Sub-View.
///
template<class T, bool isConst, class A> 
template<class BaseIterator, class ShapeIterator>
inline View<T, isConst, A>
View<T, isConst, A>::view
(
    BaseIterator bit,
    ShapeIterator sit
) const
{
    View<T, isConst, A> v;
    this->view(bit, sit, v);
    return v;
}

/// Get a sub-view.
///
/// \param bit Iterator to the beginning of a coordinate sequence
/// that determines the start position of the sub-view.
/// \param sit Iterator to the beginning of a sequence
/// that determines the shape of the sub-view.
/// \param internalCoordinateOrder Flag to set the coordinate order
/// for scalar indexing and iterators of the sub-view.
/// \return Sub-View.
///
template<class T, bool isConst, class A> 
template<class BaseIterator, class ShapeIterator>
inline View<T, isConst, A>
View<T, isConst, A>::view
(
    BaseIterator bit,
    ShapeIterator sit,
    const CoordinateOrder& internalCoordinateOrder
) const
{
    View<T, isConst, A> v;
    this->view(bit, sit, internalCoordinateOrder, v);
    return v;
}

/// Get a sub-view to constant data with the same coordinate
/// order.
///
/// \param bit Iterator to the beginning of a coordinate sequence
/// that determines the start position of the sub-view.
/// \param sit Iterator to the beginning of a sequence
/// that determines the shape of the sub-view.
/// \param out Sub-View (output).
///
template<class T, bool isConst, class A> 
template<class BaseIterator, class ShapeIterator>
inline void
View<T, isConst, A>::constView
(
    BaseIterator bit,
    ShapeIterator sit,
    View<T, true, A>& out
) const
{
    constView(bit, sit, coordinateOrder(), out);
}

/// Get a sub-view to constant data.
///
/// \param bit Iterator to the beginning of a coordinate sequence
/// that determines the start position of the sub-view.
/// \param sit Iterator to the beginning of a sequence
/// that determines the shape of the sub-view.
/// \param internalCoordinateOrder Flag to set the coordinate order
/// for scalar indexing and iterators of the sub-view. 
/// \param out Sub-View (output).
///
template<class T, bool isConst, class A> 
template<class BaseIterator, class ShapeIterator>
inline void
View<T, isConst, A>::constView
(
    BaseIterator bit,
    ShapeIterator sit,
    const CoordinateOrder& internalCoordinateOrder,
    View<T, true, A>& out
) const
{
    testInvariant();
    std::size_t offset = 0;
    coordinatesToOffset(bit, offset);
    out.assign(sit, sit+dimension(), 
        geometry_.stridesBegin(), 
        static_cast<const T*>(data_) + offset,
        internalCoordinateOrder);
}

/// Get a sub-view to constant data with the same coordinate
/// order.
///
/// \param bit Iterator to the beginning of a coordinate sequence
/// that determines the start position of the sub-view.
/// \param sit Iterator to the beginning of a sequence
/// that determines the shape of the sub-view.
/// \return Sub-View.
///
template<class T, bool isConst, class A> 
template<class BaseIterator, class ShapeIterator>
inline View<T, true, A>
View<T, isConst, A>::constView
(
    BaseIterator bit,
    ShapeIterator sit
) const
{
    View<T, true, A> v;
    this->constView(bit, sit, v);
    return v;
}

/// Get a sub-view to constant data.
///
/// \param bit Iterator to the beginning of a coordinate sequence
/// that determines the start position of the sub-view.
/// \param sit Iterator to the beginning of a sequence
/// that determines the shape of the sub-view.
/// \param internalCoordinateOrder Flag to set the coordinate order
/// for scalar indexing and iterators of the sub-view. 
/// \return Sub-View.
///
template<class T, bool isConst, class A> 
template<class BaseIterator, class ShapeIterator>
inline View<T, true, A>
View<T, isConst, A>::constView
(
    BaseIterator bit,
    ShapeIterator sit,
    const CoordinateOrder& internalCoordinateOrder
) const
{
    View<T, true, A> v;
    this->constView(bit, sit, internalCoordinateOrder, v);
    return v;
}

#ifdef HAVE_CPP11_INITIALIZER_LISTS
/// Get a sub-view.
///
/// \param b Initializer list defining the coordinate sequence
/// that determines the start position of the sub-view.
/// \param s Initializer list defining the coordinate sequence
/// that determines the stop position of the sub-view.
/// \param internalCoordinateOrder Flag to set the coordinate order
/// for scalar indexing and iterators of the sub-view.
///
template<class T, bool isConst, class A> 
inline void
View<T, isConst, A>::view
(
    std::initializer_list<std::size_t> b,
    std::initializer_list<std::size_t> s,
    const CoordinateOrder& internalCoordinateOrder,
    View<T, isConst, A>& out
) const
{
    view(b.begin(), s.begin(), internalCoordinateOrder, out);
}

/// Get a sub-view with the same coordinate order.
///
/// \param b Initializer list coordinate sequence
/// that determines the start position of the sub-view.
/// \param s Initializer list coordinate sequence
/// that determines the stop position of the sub-view.
/// \param out Sub-View (output).
///
template<class T, bool isConst, class A> 
inline void
View<T, isConst, A>::view
(
    std::initializer_list<std::size_t> b,
    std::initializer_list<std::size_t> s,
    View<T, isConst, A>& out
) const 
{
    view(b.begin(), s.begin(), coordinateOrder(), out);
}

/// Get a sub-view to constant data.
///
/// \param b Initializer list coordinate sequence
/// that determines the start position of the sub-view.
/// \param s Initializer list coordinate sequence
/// that determines the stop position of the sub-view.
/// \param internalCoordinateOrder Flag to set the coordinate order
/// for scalar indexing and iterators of the sub-view. 
///
template<class T, bool isConst, class A> 
inline void
View<T, isConst, A>::constView
(
    std::initializer_list<std::size_t> b,
    std::initializer_list<std::size_t> s,
    const CoordinateOrder& internalCoordinateOrder,
    View<T, true, A>& out
) const
{
    constView(b.begin(), s.begin(), internalCoordinateOrder, out);
}

/// Get a sub-view to constant data with the same coordinate
/// order.
///
/// \param b Initializer list coordinate sequence
/// that determines the start position of the sub-view.
/// \param s Initializer list coordinate sequence
/// that determines the stop position of the sub-view.
/// \param out Sub-View (output).
///
template<class T, bool isConst, class A>
inline void
View<T, isConst, A>::constView
(
    std::initializer_list<std::size_t> b,
    std::initializer_list<std::size_t> s,
    View<T, true, A>& out
) const
{
    constView(b.begin(), s.begin(), coordinateOrder(), out);
}
#endif

/// Reshape the View.
/// 
/// Two conditions have to be fulfilled in order for reshape to work:
/// - The new and the old shape must have the same size.
/// - The view must be simple, cf. isSimple().
/// .
/// 
/// \param begin Iterator to the beginning of a sequence that determines
/// the new shape.
/// \param end Iterator to the end of that sequence.
///
/// \sa reshapedView(), isSimple()
///
template<class T, bool isConst, class A> 
template<class ShapeIterator>
inline void
View<T, isConst, A>::reshape
(
    ShapeIterator begin,
    ShapeIterator end
)
{
    testInvariant();
    marray_detail::Assert(MARRAY_NO_DEBUG || isSimple());
    if(!MARRAY_NO_ARG_TEST) {
        std::size_t size = std::accumulate(begin, end, static_cast<std::size_t>(1), 
            std::multiplies<std::size_t>());
        marray_detail::Assert(size == this->size());
    }
    assign(begin, end, data_, coordinateOrder(), coordinateOrder());
    testInvariant();
}

/// Get a reshaped View.
/// 
/// Two conditions have to be fulfilled:
/// - The new and the old shape must have the same size.
/// - The view must be simple, cf. isSimple().
/// .
/// 
/// \param begin Iterator to the beginning of a sequence that determines
/// the new shape.
/// \param end Iterator to the end of that sequence.
///
/// \sa reshape(), isSimple()
///
template<class T, bool isConst, class A> 
template<class ShapeIterator>
inline View<T, isConst, A>
View<T, isConst, A>::reshapedView
(
    ShapeIterator begin,
    ShapeIterator end
) const
{
    View<T, isConst, A> out = *this;
    out.reshape(begin, end);
    return out;
}

#ifdef HAVE_CPP11_INITIALIZER_LISTS
/// Reshape the View.
/// 
/// Two conditions have to be fulfilled in order for reshape to work:
/// - The new and the old shape must have the same size.
/// - The view must be simple, cf. isSimple().
/// .
/// 
/// \param shape Initializer list defining the new shape.
///
/// \sa reshapedView(), isSimple()
///
template<class T, bool isConst, class A> 
inline void
View<T, isConst, A>::reshape
(
    std::initializer_list<std::size_t> shape
)
{
    reshape(shape.begin(), shape.end());
}

/// Get a reshaped View.
/// 
/// Two conditions have to be fulfilled:
/// - The new and the old shape must have the same size.
/// - The view must be simple, cf. isSimple().
/// .
/// 
/// \param shape Initializer list defining the new shape.
///
/// \sa reshape(), isSimple()
///
template<class T, bool isConst, class A> 
inline View<T, isConst, A>
View<T, isConst, A>::reshapedView
(
    std::initializer_list<std::size_t> shape
) const
{
    return reshapedView(shape.begin(), shape.end());
}
#endif

/// Get a View where one coordinate is bound to a value.
///
/// Binds one coordinate to a certain value. This reduces the
/// dimension by 1.
///
/// \param dimension Dimension of the coordinate to bind.
/// \param value Value to assign to the coordinate.
/// \return The bound view.
/// \sa squeeze(), squeezeView()
///
template<class T, bool isConst, class A> 
View<T, isConst, A>
View<T, isConst, A>::boundView
(
    const std::size_t dimension,
    const std::size_t value
) const
{
    testInvariant();
    marray_detail::Assert(MARRAY_NO_ARG_TEST || (dimension < this->dimension()
        && value < shape(dimension)));
    if(this->dimension() == 1) {
        View v(&((*this)(value)));
        v.geometry_.coordinateOrder() = coordinateOrder();
        return v;
    }
    else {
        View v;
        v.geometry_.resize(this->dimension()-1);
        v.geometry_.coordinateOrder() = coordinateOrder();
        v.geometry_.size() = size() / shape(dimension);
        for(std::size_t j=0, k=0; j<this->dimension(); ++j) {
            if(j != dimension) {
                v.geometry_.shape(k) = shape(j);
                v.geometry_.strides(k) = strides(j);
                ++k;
            }
        }
        marray_detail::stridesFromShape(v.geometry_.shapeBegin(), v.geometry_.shapeEnd(),
            v.geometry_.shapeStridesBegin(), v.geometry_.coordinateOrder());
        v.data_ = data_ + strides(dimension) * value;
        v.updateSimplicity();
        v.testInvariant();
        return v;
    }
}

/// Remove singleton dimensions by setting their coordinates to zero.
///
/// \sa squeezedView(), boundView()
///
template<class T, bool isConst, class A> 
void
View<T, isConst, A>::squeeze()
{
    testInvariant();
    if(dimension() != 0) {
        std::size_t newDimension = dimension();
        for(std::size_t j=0; j<dimension(); ++j) {
            if(shape(j) == 1) {
                --newDimension;
            }
        }
        if(newDimension != dimension()) {
            if(newDimension == 0) {
                geometry_.resize(0);
                geometry_.size() = 1;
            }
            else {
                for(std::size_t j=0, k=0; j<geometry_.dimension(); ++j) {
                    if(geometry_.shape(j) != 1) {
                        geometry_.shape(k) = geometry_.shape(j);
                        geometry_.strides(k) = geometry_.strides(j);
                        ++k;
                    }
                }
                geometry_.resize(newDimension);
                marray_detail::stridesFromShape(geometry_.shapeBegin(), geometry_.shapeEnd(), 
                    geometry_.shapeStridesBegin(), geometry_.coordinateOrder());
                updateSimplicity();
            }
        }
    }
    testInvariant();
}

/// Get a View where all singleton dimensions have been removed by
/// setting their coordinates to zero.
///
/// \sa squeeze(), boundView()
///
template<class T, bool isConst, class A> 
inline View<T, isConst, A>
View<T, isConst, A>::squeezedView() const
{
    View<T, isConst, A> v = *this;
    v.squeeze();
    return v;
}

#ifdef HAVE_CPP11_INITIALIZER_LISTS
/// Permute dimensions.
///
/// \param begin Iterator to the beginning of a sequence which
/// has to contain the integers 0, ..., dimension()-1 in any
/// order. Otherwise, a runtime error is thrown.
/// \sa permutedView(), transpose(), transposedView(), shift(),
/// shiftedView()
///
template<class T, bool isConst, class A>
void
View<T, isConst, A>::permute
(
    std::initializer_list<std::size_t> permutation
)
{
    permute(permutation.begin());
}
#endif

/// Permute dimensions.
///
/// \param begin Iterator to the beginning of a sequence which
/// has to contain the integers 0, ..., dimension()-1 in any
/// order. Otherwise, a runtime error is thrown.
/// \sa permutedView(), transpose(), transposedView(), shift(),
/// shiftedView()
///
template<class T, bool isConst, class A> 
template<class CoordinateIterator>
void
View<T, isConst, A>::permute
(
    CoordinateIterator begin
)
{
    testInvariant();
    if(!MARRAY_NO_ARG_TEST) {
        marray_detail::Assert(dimension() != 0);
        std::set<std::size_t> s1, s2;
        CoordinateIterator it = begin;
        for(std::size_t j=0; j<dimension(); ++j) {
            s1.insert(j);
            s2.insert(*it);
            ++it;
        }
        marray_detail::Assert(s1 == s2);
    }
    // update shape, shape strides, strides, and simplicity
    std::vector<std::size_t> newShape = std::vector<std::size_t>(dimension());
    std::vector<std::size_t> newStrides = std::vector<std::size_t>(dimension());
    for(std::size_t j=0; j<dimension(); ++j) {
        newShape[j] = geometry_.shape(static_cast<std::size_t>(*begin));
        newStrides[j] = geometry_.strides(static_cast<std::size_t>(*begin));
        ++begin;
    }
    for(std::size_t j=0; j<dimension(); ++j) {
        geometry_.shape(j) = newShape[j];
        geometry_.strides(j) = newStrides[j];
    }
    marray_detail::stridesFromShape(geometry_.shapeBegin(), geometry_.shapeEnd(),
        geometry_.shapeStridesBegin(), geometry_.coordinateOrder());
    updateSimplicity();
    testInvariant();
}

/// Get a View with permuted dimensions.
///
/// \param begin Iterator to the beginning of a sequence which
/// has to contain the integers 0, ..., dimension()-1 in any
/// order. Otherwise, a runtime error is thrown.
/// \return Permuted View.
/// \sa permute(), transpose(), transposedView(), shift(),
/// shiftedView()
///
template<class T, bool isConst, class A> 
template<class CoordinateIterator>
inline View<T, isConst, A>
View<T, isConst, A>::permutedView
(
    CoordinateIterator begin
) const
{
    View<T, isConst, A> out = *this;
    out.permute(begin);
    return out;
}

/// Exchange two dimensions.
///
/// \param c1 Dimension
/// \param c2 Dimension
/// \sa permute(), permutedView(), shift(), shiftedView()
///
template<class T, bool isConst, class A> 
void
View<T, isConst, A>::transpose
(
    const std::size_t c1,
    const std::size_t c2
)
{
    testInvariant();
    marray_detail::Assert(MARRAY_NO_ARG_TEST ||
        (dimension() != 0 && c1 < dimension() && c2 < dimension()));

    std::size_t j1 = c1;
    std::size_t j2 = c2;
    std::size_t c;
    std::size_t d;

    // transpose shape
    c = geometry_.shape(j2);
    geometry_.shape(j2) = geometry_.shape(j1);
    geometry_.shape(j1) = c;

    // transpose strides
    d = geometry_.strides(j2);
    geometry_.strides(j2) = geometry_.strides(j1);
    geometry_.strides(j1) = d;

    // update shape strides
    marray_detail::stridesFromShape(geometry_.shapeBegin(), geometry_.shapeEnd(),
        geometry_.shapeStridesBegin(), geometry_.coordinateOrder());

    updateSimplicity();
    testInvariant();
}

/// Reverse dimensions.
///
/// \sa transposedView(), permute(), permutedView(), shift(),
/// shiftedView()
///
template<class T, bool isConst, class A> 
void
View<T, isConst, A>::transpose()
{
    testInvariant();
    for(std::size_t j=0; j<dimension()/2; ++j) {
        std::size_t k = dimension()-j-1;

        // transpose shape
        std::size_t tmp = geometry_.shape(j);
        geometry_.shape(j) = geometry_.shape(k);
        geometry_.shape(k) = tmp;

        // transpose strides
        tmp = geometry_.strides(j);
        geometry_.strides(j) = geometry_.strides(k);
        geometry_.strides(k) = tmp;
    }
    marray_detail::stridesFromShape(geometry_.shapeBegin(), geometry_.shapeEnd(),
        geometry_.shapeStridesBegin(), geometry_.coordinateOrder());
    updateSimplicity();
    testInvariant();
}

/// Get a View with two dimensions exchanged.
///
/// \param c1 Dimension
/// \param c2 Dimension
/// \return Transposed View.
/// \sa transpose(), permute(), permutedView(), shift(),
/// shiftedView()
///
template<class T, bool isConst, class A> 
inline View<T, isConst, A>
View<T, isConst, A>::transposedView
(
    const std::size_t c1,
    const std::size_t c2
) const
{
    View<T, isConst, A> out = *this;
    out.transpose(c1, c2);
    return out;
}

/// Get a View with dimensions reversed.
///
/// \return View with dimensions reversed.
/// \sa transpose(), permute(), permutedView(), shift(),
/// shiftedView()
///
template<class T, bool isConst, class A> 
inline View<T, isConst, A>
View<T, isConst, A>::transposedView() const
{
    View<T, isConst, A> out = *this;
    out.transpose();
    return out;
}

/// Cycle shift dimensions.
///
/// \param n Number of positions to shift
/// \sa shiftedView(), permute(), permutedView(), transpose(),
/// transposedView()
///
template<class T, bool isConst, class A> 
inline void
View<T, isConst, A>::shift
(
    const int n
) 
{
    testInvariant();
    marray_detail::Assert(MARRAY_NO_DEBUG || dimension() != 0);
    if(n <= -static_cast<int>(dimension()) || n >= static_cast<int>(dimension())) {
        shift(n % static_cast<int>(dimension()));
    }
    else {
        if(n > 0) {
            shift(n - static_cast<int>(dimension()));
        }
        else {
            std::vector<std::size_t> p(dimension());
            for(std::size_t j=0; j<dimension(); ++j) {
                p[j] = static_cast<std::size_t>((static_cast<int>(j) - n)) % dimension();
            }
            permute(p.begin());
        }
    }
    testInvariant();
}

/// Get a View which dimensions cycle shifted.
///
/// \param n Number of positions to shift
/// \sa shift(), permute(), permutedView(), transpose(), transposedView()
///
template<class T, bool isConst, class A> 
inline View<T, isConst, A>
View<T, isConst, A>::shiftedView
(
    const int n
) const
{
    View<T, isConst, A> out = *this;
    out.shift(n);
    return out;
}

/// Get an iterator to the beginning.
///
/// \return Iterator.
/// \sa end()
///

template<class T, bool isConst, class A> 
inline typename View<T, isConst, A>::iterator
View<T, isConst, A>::begin()
{
    testInvariant();
    return Iterator<T, isConst, A>(*this, 0);
}

/// Get the end-iterator.
///
/// \return Iterator.
/// \sa begin()
///
template<class T, bool isConst, class A> 
inline typename View<T, isConst, A>::iterator
View<T, isConst, A>::end()
{
    testInvariant();
    return Iterator<T, isConst, A>(*this, geometry_.size());
}

/// Get an iterator to the beginning.
///
/// \return Iterator.
/// \sa end()
///
template<class T, bool isConst, class A> 
inline typename View<T, isConst, A>::const_iterator
View<T, isConst, A>::begin() const
{
    testInvariant();
    return Iterator<T, true>(*this, 0);
}

/// Get the end-iterator.
///
/// \return Iterator.
/// \sa begin()
///
template<class T, bool isConst, class A> 
inline typename View<T, isConst, A>::const_iterator

View<T, isConst, A>::end() const
{
    testInvariant();
    return Iterator<T, true>(*this, geometry_.size());
}

/// Get a reserve iterator to the beginning.
///
/// \return Iterator.
/// \sa rend()
///
template<class T, bool isConst, class A> 
inline typename View<T, isConst, A>::reverse_iterator
View<T, isConst, A>::rbegin()
{
    return reverse_iterator(end());
}

/// Get the reverse end-iterator.
///
/// \return Iterator.
/// \sa rbegin()
///
template<class T, bool isConst, class A> 
inline typename View<T, isConst, A>::reverse_iterator
View<T, isConst, A>::rend()
{
    return reverse_iterator(begin());
}

/// Get a reserve iterator to the beginning.
///
/// \return Iterator.
/// \sa rend()
///
template<class T, bool isConst, class A> 
inline typename View<T, isConst, A>::const_reverse_iterator
View<T, isConst, A>::rbegin() const
{
    return const_reverse_iterator(end());
}

/// Get the reverse end-iterator.
///
/// \return Iterator.
/// \sa rbegin()
///
template<class T, bool isConst, class A> 
inline typename View<T, isConst, A>::const_reverse_iterator
View<T, isConst, A>::rend() const
{
    return const_reverse_iterator(begin());
}

/// Update Simplicity.
///
/// This function sets the redundant boolean attribute isSimple_.
/// isSimple_ is set to true if the shape strides equal the 
/// strides. 
///
template<class T, bool isConst, class A> 
inline void
View<T, isConst, A>::updateSimplicity()
{
    // no invariant test here because this function 
    // is called during unsafe updates of a view
    geometry_.updateSimplicity();
}

/// Unsafe direct memory access.
///
/// This function provides direct access to the data items under the view 
/// in the order in which these items reside in memory.
///
/// \param offset offset to be added to the data pointer.
/// \return constant reference to the data item.
///
template<class T, bool isConst, class A> 
inline const T& 
View<T, isConst, A>::operator[]
(
    const std::size_t offset
) 
const
{
    return data_[offset];
}

/// Unsafe direct memory access.
///
/// This function provides direct access to the data items under the view 
/// in the order in which these items reside in memory.
///
/// \param offset offset to be added to the data pointer.
/// \return reference to the data item.
///
template<class T, bool isConst, class A> 
inline T& 
View<T, isConst, A>::operator[]
(
    const std::size_t offset
)
{
    return data_[offset];
}

/// Test invariant.
///
/// This function tests the invariant of View and thus the consistency
/// of redundant information.
///
template<class T, bool isConst, class A> 
inline void
View<T, isConst, A>::testInvariant() const
{
    if(!MARRAY_NO_DEBUG) {
        if(geometry_.dimension() == 0) {
            marray_detail::Assert(geometry_.isSimple() == true);
            if(data_ != 0) { // scalar
                marray_detail::Assert(geometry_.size() == 1);
            }
        }
        else {
            marray_detail::Assert(data_ != 0);

            // test size_ to be consistent with shape_
            std::size_t testSize = 1;
            for(std::size_t j=0; j<geometry_.dimension(); ++j) {
                testSize *= geometry_.shape(j);
            }
            marray_detail::Assert(geometry_.size() == testSize);

            // test shapeStrides_ to be consistent with shape_
            if(geometry_.coordinateOrder() == FirstMajorOrder) {
                std::size_t tmp = 1;
                for(std::size_t j=0; j<geometry_.dimension(); ++j) {
                    marray_detail::Assert(geometry_.shapeStrides(geometry_.dimension()-j-1) == tmp);
                    tmp *= geometry_.shape(geometry_.dimension()-j-1);
                }
            }
            else {
                std::size_t tmp = 1;
                for(std::size_t j=0; j<geometry_.dimension(); ++j) {
                    marray_detail::Assert(geometry_.shapeStrides(j) == tmp);
                    tmp *= geometry_.shape(j);
                }
            }

            // test the simplicity condition 
            if(geometry_.isSimple()) {
                for(std::size_t j=0; j<geometry_.dimension(); ++j) {
                    marray_detail::Assert(geometry_.strides(j) == geometry_.shapeStrides(j));
                }
            }
        }
    }
}

/// Check whether two Views overlap.
///
/// This function returns true if two memory intervals overlap:
/// (1) the interval between the first and the last element of the object
/// whose member function overlaps() is called.
/// (2) the interval between the first and the last element of v.
///
/// Note that this not necessarily implies the existence of an element 
/// that is addressed by both v and the current object. v could for
/// instance address all odd elements in a vector while the current
/// object addresses all even elements. 
///
/// \param v A view to compare with *this.
/// \return bool.
///
template<class T, bool isConst, class A> 
template<class TLocal, bool isConstLocal, class ALocal>
inline bool View<T, isConst, A>::overlaps
(
    const View<TLocal, isConstLocal, ALocal>& v
) const
{
    testInvariant();
    if(!MARRAY_NO_ARG_TEST) {
        v.testInvariant();
    }
    if(data_ == 0 || v.data_ == 0) {
        return false;
    }
    else {
        const void* dataPointer_ = data_;
        const void* vDataPointer_ = v.data_;
        const void* maxPointer = & (*this)(this->size()-1);
        const void* maxPointerV = & v(v.size()-1);
        if(    (dataPointer_   <= vDataPointer_ && vDataPointer_ <= maxPointer)
            || (vDataPointer_ <= dataPointer_   && dataPointer_   <= maxPointerV) )
        {
            return true;
        }
    }
    return false;
}

/// Output as string.
///
template<class T, bool isConst, class A>
std::string 
View<T, isConst, A>::asString
(
    const StringStyle& style
) const
{
    testInvariant();
    std::ostringstream out(std::ostringstream::out);
    if(style == MatrixStyle) {
        if(dimension() == 0) {
            // scalar
            out << "A = " << (*this)(0) << std::endl;
        }
        else if(dimension() == 1) {
            // vector
            out << "A = (";
            for(std::size_t j=0; j<this->size(); ++j) {
                out << (*this)(j) << ", ";
            }
            out << "\b\b)" << std::endl;
        }
        else if(dimension() == 2) {
            // matrix
            if(coordinateOrder() == FirstMajorOrder) {
                out << "A(r,c) =" << std::endl;
                for(std::size_t y=0; y<this->shape(0); ++y) {
                    for(std::size_t x=0; x<this->shape(1); ++x) {
                        out << (*this)(y, x) << ' ';
                    }
                    out << std::endl;
                }
            }
            else {
                out << "A(c,r) =" << std::endl;
                for(std::size_t y=0; y<this->shape(1); ++y) {
                    for(std::size_t x=0; x<this->shape(0); ++x) {
                        out << (*this)(x, y) << ' ';
                    }
                    out << std::endl;
                }
            }
        }
        else {
            // higher dimensional
            std::vector<std::size_t> c1(dimension());
            std::vector<std::size_t> c2(dimension());
            unsigned short q = 2;
            if(coordinateOrder() == FirstMajorOrder) {
                q = static_cast<unsigned char>(dimension() - 3);
            }
            for(const_iterator it = this->begin(); it.hasMore(); ++it) {
                it.coordinate(c2.begin());
                if(it.index() == 0 || c2[q] != c1[q]) {
                    if(it.index() != 0) {
                        out << std::endl << std::endl;
                    }
                    if(coordinateOrder() == FirstMajorOrder) {
                        out << "A(";
                        for(std::size_t j=0; j<dimension()-2; ++j) {
                            out << c2[j] << ",";
                        }
                    }
                    else {
                        out << "A(c,r,";
                        for(std::size_t j=2; j<dimension(); ++j) {
                            out << c2[j] << ",";
                        }
                    }
                    out << '\b';
                    if(coordinateOrder() == FirstMajorOrder) {
                        out << ",r,c";
                    }
                    out << ") =" << std::endl;
                }
                else if(c2[1] != c1[1]) {
                    out << std::endl;
                }
                out << *it << " ";
                c1 = c2;
            }
            out << std::endl;
        }
        out << std::endl;
    }
    else if(style == TableStyle) {
        if(dimension() == 0) {
            // scalar
            out << "A = " << (*this)(0) << std::endl;
        }
        else {
            // non-scalar
            std::vector<std::size_t> c(dimension());
            for(const_iterator it = this->begin(); it.hasMore(); ++it) {
                out << "A(";
                it.coordinate(c.begin());
                for(std::size_t j=0; j<c.size(); ++j) {
                    out << c[j] << ',';
                }
                out << "\b) = " << *it << std::endl;
            }
        }
        out << std::endl;
    }
    return out.str();
}

// implementation of arithmetic operators of View

template<class T1, class T2, bool isConst, class A>
inline View<T1, false, A>&
operator+=
(
    View<T1, false, A>& v,
    const View<T2, isConst, A>& w
)
{
    marray_detail::operate(v, w, marray_detail::PlusEqual<T1, T2>());
    return v;
}

// prefix
template<class T, class A>
inline View<T, false, A>&
operator++
(
    View<T, false, A>& v
)
{
    marray_detail::operate(v, marray_detail::PrefixIncrement<T>());
    return v;
}

// postfix
template<class T, class A>
inline Marray<T, A>
operator++
(
    Marray<T, A>& in,
    int dummy
) 
{
    Marray<T, A> out = in; 
    marray_detail::operate(in, marray_detail::PostfixIncrement<T>());
    return out;
}

template<class T1, class T2, bool isConst, class A>
inline View<T1, false, A>&
operator-=
(
    View<T1, false, A>& v,
    const View<T2, isConst, A>& w
)
{
    marray_detail::operate(v, w, marray_detail::MinusEqual<T1, T2>());
    return v;
}

// prefix
template<class T, class A>
inline View<T, false, A>&
operator--
(
    View<T, false, A>& v
)
{
    marray_detail::operate(v, marray_detail::PrefixDecrement<T>());
    return v;
}

// postfix
template<class T, class A>
inline Marray<T, A>
operator--
(
    Marray<T, A>& in,
    int dummy
) 
{
    Marray<T, A> out = in; 
    marray_detail::operate(in, marray_detail::PostfixDecrement<T>());
    return out;
}

template<class T1, class T2, bool isConst, class A>
inline View<T1, false, A>&
operator*=
(
    View<T1, false, A>& v,
    const View<T2, isConst, A>& w
)
{
    marray_detail::operate(v, w, marray_detail::TimesEqual<T1, T2>());
    return v;
}

template<class T1, class T2, bool isConst, class A>
inline View<T1, false, A>&
operator/=
(
    View<T1, false, A>& v,
    const View<T2, isConst, A>& w
)
{
    marray_detail::operate(v, w, marray_detail::DividedByEqual<T1, T2>());
    return v;
}

template<class E1, class T1, class E2, class T2>
inline const BinaryViewExpression<E1, T1, E2, T2, marray_detail::Plus<T1, T2, typename marray_detail::PromoteType<T1, T2>::type> >
operator+
(
    const ViewExpression<E1, T1>& expression1, 
    const ViewExpression<E2, T2>& expression2
)
{
    typedef typename marray_detail::PromoteType<T1, T2>::type promoted_type;
    typedef marray_detail::Plus<T1, T2, promoted_type> Functor;
    typedef BinaryViewExpression<E1, T1, E2, T2, Functor> return_type; 
    return return_type(expression1, expression2);
}

template<class E, class T>
inline const ViewExpression<E,T>&
operator+
(
    const ViewExpression<E,T>& expression
) // unary
{
    return expression;
}

#define MARRAY_UNARY_OPERATOR(datatype, operation, functorname) \
template<class T, class A> \
inline View<T, false, A>& \
operator operation \
( \
    View<T, false, A>& v, \
    const datatype& x \
) \
{ \
    marray_detail::operate(v, static_cast<T>(x), marray_detail:: functorname <T, T>()); \
    return v; \
} \

#define MARRAY_UNARY_OPERATOR_ALL_TYPES(op, functorname) \
    MARRAY_UNARY_OPERATOR(char, op, functorname) \
    MARRAY_UNARY_OPERATOR(unsigned char, op, functorname) \
    MARRAY_UNARY_OPERATOR(short, op, functorname) \
    MARRAY_UNARY_OPERATOR(unsigned short, op, functorname) \
    MARRAY_UNARY_OPERATOR(int, op, functorname) \
    MARRAY_UNARY_OPERATOR(unsigned int, op, functorname) \
    MARRAY_UNARY_OPERATOR(long, op, functorname) \
    MARRAY_UNARY_OPERATOR(unsigned long, op, functorname) \
    MARRAY_UNARY_OPERATOR(float, op, functorname) \
    MARRAY_UNARY_OPERATOR(double, op, functorname) \
    MARRAY_UNARY_OPERATOR(long double, op, functorname) \

MARRAY_UNARY_OPERATOR_ALL_TYPES(+=, PlusEqual)
MARRAY_UNARY_OPERATOR_ALL_TYPES(-=, MinusEqual)
MARRAY_UNARY_OPERATOR_ALL_TYPES(*=, TimesEqual)
MARRAY_UNARY_OPERATOR_ALL_TYPES(/=, DividedByEqual)

template<class E1, class T1, class E2, class T2>
inline const BinaryViewExpression<E1, T1, E2, T2,
    marray_detail::Minus<T1, T2, typename marray_detail::PromoteType<T1, T2>::type> >
operator-
(
    const ViewExpression<E1, T1>& expression1, 
    const ViewExpression<E2, T2>& expression2
)
{
    return BinaryViewExpression<E1, T1, E2, T2,
        marray_detail::Minus<T1, T2, 
            typename marray_detail::PromoteType<T1, T2>::type> >(
            expression1, expression2);
}

template<class E, class T>
inline const UnaryViewExpression<E,T,marray_detail::Negate<T> >
operator-
(
    const ViewExpression<E,T>& expression
) // unary
{
    return UnaryViewExpression<E,T,marray_detail::Negate<T> >(
        expression);
}

template<class E1, class T1, class E2, class T2>
inline const BinaryViewExpression<E1, T1, E2, T2,
    marray_detail::Times<T1, T2, typename marray_detail::PromoteType<T1, T2>::type> >
operator*
(
    const ViewExpression<E1, T1>& expression1, 
    const ViewExpression<E2, T2>& expression2
)
{
    return BinaryViewExpression<E1, T1, E2, T2,
        marray_detail::Times<T1, T2, 
            typename marray_detail::PromoteType<T1, T2>::type > >(
            expression1, expression2);
}

template<class E1, class T1, class E2, class T2>
inline const BinaryViewExpression<E1, T1, E2, T2,
    marray_detail::DividedBy<T1, T2, typename marray_detail::PromoteType<T1, T2>::type> >
operator/
(
    const ViewExpression<E1, T1>& expression1, 
    const ViewExpression<E2, T2>& expression2
)
{
    return BinaryViewExpression<E1, T1, E2, T2,
        marray_detail::DividedBy<T1, T2, 
            typename marray_detail::PromoteType<T1, T2>::type > >(
            expression1, expression2);
}

#define MARRAY_BINARY_OPERATOR(datatype, operation, functorname) \
template<class E, class T> \
inline const BinaryViewExpressionScalarSecond< \
    E, T, datatype, marray_detail:: functorname < \
        T, datatype, typename marray_detail::PromoteType<T, datatype>::type \
    > \
> \
operator operation \
( \
    const ViewExpression<E, T>& expression, \
    const datatype& scalar \
) \
{ \
    typedef typename marray_detail::PromoteType<T, datatype>::type \
        promoted_type; \
    typedef marray_detail:: functorname <T, datatype, promoted_type> Functor; \
    typedef BinaryViewExpressionScalarSecond<E, T, datatype, Functor> \
        expression_type; \
    return expression_type(expression, scalar); \
} \
\
template<class E, class T> \
inline const BinaryViewExpressionScalarFirst \
< \
    E, T, datatype, marray_detail:: functorname < \
        datatype, T, typename marray_detail::PromoteType<datatype, T>::type \
    > \
> \
operator operation \
( \
    const datatype& scalar, \
    const ViewExpression<E, T>& expression \
) \
{ \
    typedef typename marray_detail::PromoteType<T, datatype>::type \
        promoted_type; \
    typedef marray_detail:: functorname <datatype, T, promoted_type> Functor; \
    typedef BinaryViewExpressionScalarFirst<E, T, datatype, Functor> \
        expression_type; \
    return expression_type(expression, scalar); \
}

#define MARRAY_BINARY_OPERATOR_ALL_TYPES(op, functorname) \
    MARRAY_BINARY_OPERATOR(char, op, functorname) \
    MARRAY_BINARY_OPERATOR(unsigned char, op, functorname) \
    MARRAY_BINARY_OPERATOR(short, op, functorname) \
    MARRAY_BINARY_OPERATOR(unsigned short, op, functorname) \
    MARRAY_BINARY_OPERATOR(int, op, functorname) \
    MARRAY_BINARY_OPERATOR(unsigned int, op, functorname) \
    MARRAY_BINARY_OPERATOR(long, op, functorname) \
    MARRAY_BINARY_OPERATOR(unsigned long, op, functorname) \
    MARRAY_BINARY_OPERATOR(float, op, functorname) \
    MARRAY_BINARY_OPERATOR(double, op, functorname) \
    MARRAY_BINARY_OPERATOR(long double, op, functorname) \

MARRAY_BINARY_OPERATOR_ALL_TYPES(+, Plus)
MARRAY_BINARY_OPERATOR_ALL_TYPES(-, Minus)
MARRAY_BINARY_OPERATOR_ALL_TYPES(*, Times)
MARRAY_BINARY_OPERATOR_ALL_TYPES(/, DividedBy)

// implementation of Marray

/// Clear Marray.
///
/// Leaves the Marray in the same state as if the empty constructor
/// had been called. Previously allocated memory is de-allocated.
///
/// \param allocator Allocator.
/// \sa Marray()
///
template<class T, class A> 
inline void
Marray<T, A>::assign
(
    const allocator_type& allocator
)
{
    if(this->data_ != 0) {
        dataAllocator_.deallocate(this->data_, this->size());
        this->data_ = 0;
    }
    dataAllocator_ = allocator;
    base::assign();
}

/// Empty constructor.
///
/// \param allocator Allocator. 
///
template<class T, class A> 
inline
Marray<T, A>::Marray
(
    const allocator_type& allocator
) 
: base(allocator),
  dataAllocator_(allocator)
{
    testInvariant();
}

/// Construct 0-dimensional (scalar) array.
///
/// \param value Value of the single data item.
/// \param coordinateOrder Flag specifying whether FirstMajorOrder or
/// LastMajorOrder is to be used. As the Marray can be resized after 
/// construction, the coordinate order has to be set even for a
/// 0-dimensional Marray.
/// \param allocator Allocator.
///
template<class T, class A> 
inline
Marray<T, A>::Marray
(
    const T& value,
    const CoordinateOrder& coordinateOrder,
    const allocator_type& allocator
) 
:   dataAllocator_(allocator)
{
    this->data_ = dataAllocator_.allocate(1);
    this->data_[0] = value;
    this->geometry_ = geometry_type(0, coordinateOrder, 1, true, allocator);
    testInvariant();
}

/// Copy from a Marray.
///
/// \param in Marray (source).
///
template<class T, class A> 
inline
Marray<T, A>::Marray
(
    const Marray<T, A>& in
)
:   dataAllocator_(in.dataAllocator_) 
{
    if(!MARRAY_NO_ARG_TEST) {
        in.testInvariant();
    }
    if(in.data_ == 0) {
        this->data_ = 0;
    }
    else {
        this->data_ = dataAllocator_.allocate(in.size());
        memcpy(this->data_, in.data_, (in.size())*sizeof(T));
    }
    this->geometry_ = in.geometry_;
    testInvariant();
}

/// Copy from a View.
///
/// \param in View (source).
///
template<class T, class A> 
template<class TLocal, bool isConstLocal, class ALocal>
inline
Marray<T, A>::Marray
(
    const View<TLocal, isConstLocal, ALocal>& in
) 
: dataAllocator_()
{
    if(!MARRAY_NO_ARG_TEST) {
        in.testInvariant();
    }

    // adapt geometry
    this->geometry_ = in.geometry_;
    for(std::size_t j=0; j<in.dimension(); ++j) {
        this->geometry_.strides(j) = in.geometry_.shapeStrides(j); // !
    }
    this->geometry_.isSimple() = true;

    // copy data
    if(in.size() == 0) {
        this->data_ = 0;
    }
    else {
        this->data_ = dataAllocator_.allocate(in.size());
    }
    if(in.isSimple() && marray_detail::IsEqual<T, TLocal>::type) {
        memcpy(this->data_, in.data_, (in.size())*sizeof(T));
    }
    else {
        typename View<TLocal, isConstLocal, ALocal>::const_iterator it = in.begin();
        for(std::size_t j=0; j<this->size(); ++j, ++it)  {
            this->data_[j] = static_cast<T>(*it);
        }
    }

    testInvariant();
}

/// Construct Marray from ViewExpression.
///
/// \param expression ViewExpression.
/// \param allocator Allocator.
///
template<class T, class A> 
template<class E, class Te>
inline
Marray<T, A>::Marray
(
    const ViewExpression<E, Te>& expression,
    const allocator_type& allocator
) 
:   dataAllocator_(allocator)
{
    this->data_ = dataAllocator_.allocate(expression.size());
    if(expression.dimension() == 0) {
        this->geometry_ = geometry_type(0, 
            static_cast<const E&>(expression).coordinateOrder(), 
            1, true, dataAllocator_);
    }
    else {
        this->geometry_ = geometry_type(
            static_cast<const E&>(expression).shapeBegin(), 
            static_cast<const E&>(expression).shapeEnd(),
            static_cast<const E&>(expression).coordinateOrder(),
            static_cast<const E&>(expression).coordinateOrder(),
            dataAllocator_);

    }
    const E& e = static_cast<const E&>(expression);
    if(e.dimension() == 0) {
        marray_detail::Assert(MARRAY_NO_ARG_TEST || e.size() < 2);
        this->data_[0] = static_cast<T>(e(0));
    }
    else {
        marray_detail::Assert(MARRAY_NO_ARG_TEST || e.size() != 0);
        marray_detail::operate(*this, e, marray_detail::Assign<T, Te>());
    }
    testInvariant();
}

/// Construct Marray with initialization.
///
/// \param begin Iterator to the beginning of a sequence that determines
/// the shape.
/// \param end Iterator to the end of that sequence.
/// \param value Value with which all entries are initialized.
/// \param coordinateOrder Flag specifying whether FirstMajorOrder or
/// LastMajorOrder is to be used.
/// \param allocator Allocator.
///
template<class T, class A> 
template<class ShapeIterator>
inline
Marray<T, A>::Marray
(
    ShapeIterator begin,
    ShapeIterator end,
    const T& value,
    const CoordinateOrder& coordinateOrder,
    const allocator_type& allocator
)
: dataAllocator_(allocator)
{
    std::size_t size = std::accumulate(begin, end, static_cast<std::size_t>(1), 
        std::multiplies<std::size_t>());
    marray_detail::Assert(MARRAY_NO_ARG_TEST || size != 0);
    base::assign(begin, end, dataAllocator_.allocate(size), coordinateOrder, 
        coordinateOrder, allocator); 
    for(std::size_t j=0; j<size; ++j) {
        this->data_[j] = value;
    }
    testInvariant();
}

/// Construct Marray without initialization.
///
/// \param is Flag to be set to SkipInitialization.
/// \param begin Iterator to the beginning of a sequence that determines
/// the shape.
/// \param end Iterator to the end of that sequence.
/// \param coordinateOrder Flag specifying whether FirstMajorOrder or
/// LastMajorOrder is to be used.
/// \param allocator Allocator.
///
template<class T, class A> 
template<class ShapeIterator>
inline
Marray<T, A>::Marray
(
    const InitializationSkipping& is,
    ShapeIterator begin,
    ShapeIterator end,
    const CoordinateOrder& coordinateOrder,
    const allocator_type& allocator
) 
: dataAllocator_(allocator)
{
    std::size_t size = std::accumulate(begin, end, static_cast<std::size_t>(1), 
        std::multiplies<std::size_t>());
    marray_detail::Assert(MARRAY_NO_ARG_TEST || size != 0);
    base::assign(begin, end, dataAllocator_.allocate(size), coordinateOrder, 
        coordinateOrder, allocator); 
    testInvariant();
}

#ifdef HAVE_CPP11_INITIALIZER_LISTS
/// Construct Marray with initialization.
///
/// \param begin Shape given as initializer list.
/// \param value Value with which all entries are initialized.
/// \param coordinateOrder Flag specifying whether FirstMajorOrder or
/// LastMajorOrder is to be used.
/// \param allocator Allocator.
///
template<class T, class A>
inline
Marray<T, A>::Marray
(
    std::initializer_list<std::size_t> shape,
    const T& value,
    const CoordinateOrder& coordinateOrder,
    const allocator_type& allocator

) 
: dataAllocator_(allocator)
{
    std::size_t size = std::accumulate(shape.begin(), shape.end(), 
        static_cast<std::size_t>(1), std::multiplies<std::size_t>());
    marray_detail::Assert(MARRAY_NO_ARG_TEST || size != 0);
    base::assign(shape.begin(), shape.end(), dataAllocator_.allocate(size), 
                 coordinateOrder, coordinateOrder, allocator); 
    std::fill(this->data_, this->data_+size, value);
    testInvariant();
}
#endif

/// Destructor.
///
template<class T, class A> 
inline
Marray<T, A>::~Marray()
{
    dataAllocator_.deallocate(this->data_, this->size());
}

/// Assignment.
/// 
/// This operator works as follows:
/// - It always attempts to copy the data from 'in'.
/// - If 'in' and *this have the same size, already allocated memory 
///   is re-used. Otherwise, the memory allocated for *this is freed, 
///   and new memory is allocated to take the copy of 'in'.
/// - If 'in' is un-initialized, memory allocated for *this is freed.
/// .
///
/// \param in Marray (source).
/// 
template<class T, class A> 

Marray<T, A>&
Marray<T, A>::operator=
(
    const Marray<T, A>& in
)
{
    testInvariant();
    if(!MARRAY_NO_ARG_TEST) {
        in.testInvariant();
    }
    if(this != &in) { // no self-assignment
        // copy data
        if(in.data_ == 0) { // un-initialized
            // free
            dataAllocator_.deallocate(this->data_, this->size());
            this->data_ = 0;
        }
        else {
            if(this->size() != in.size()) {
                // re-alloc
                dataAllocator_.deallocate(this->data_, this->size());
                this->data_ = dataAllocator_.allocate(in.size());
            }
            // copy data
            memcpy(this->data_, in.data_, in.size() * sizeof(T));
        }
        this->geometry_ = in.geometry_;
    }
    testInvariant();
    return *this;
}

/// Assignment from View.
///
/// This operator works as follows:
/// - It always attempts to copy the data from 'in'.
/// - If 'in' and *this have overlap, a copy of 'in' is made and 
///   assigned to *this.
/// - If 'in' and *this have the same size, already allocated memory 
///   is re-used. Otherwise, the memory allocated for *this is freed, 
///   and new memory is allocated to take the copy of 'in'.
/// - If 'in' is un-initialized, memory allocated for *this is freed.
/// .
/// 
/// \param in View (source).
/// 
template<class T, class A> 
template<class TLocal, bool isConstLocal, class ALocal>
Marray<T, A>&
Marray<T, A>::operator=
(
    const View<TLocal, isConstLocal, ALocal>& in
)
{
    if(!MARRAY_NO_ARG_TEST) {
        in.testInvariant();
    }
    if( (void*)(this) != (void*)(&in) ) { // no self-assignment
        if(in.data_ == 0) {
            dataAllocator_.deallocate(this->data_, this->size());
            this->data_ = 0;
            this->geometry_ = in.geometry_;
        }
        else if(this->overlaps(in)) {
            Marray<T, A> m = in; // temporary copy
            (*this) = m;
        }
        else {
            // re-alloc memory if necessary
            if(this->size() != in.size()) {
                dataAllocator_.deallocate(this->data_, this->size());
                this->data_ = dataAllocator_.allocate(in.size());
            }

            // copy geometry
            this->geometry_.resize(in.dimension());
            for(std::size_t j=0; j<in.dimension(); ++j) {
                this->geometry_.shape(j) = in.geometry_.shape(j);
                this->geometry_.shapeStrides(j) = in.geometry_.shapeStrides(j);
                this->geometry_.strides(j) = in.geometry_.shapeStrides(j); // !
            }
            this->geometry_.size() = in.size();
            this->geometry_.isSimple() = true;
            this->geometry_.coordinateOrder() = in.coordinateOrder();

            // copy data
            if(in.isSimple() && marray_detail::IsEqual<T, TLocal>::type) {
                memcpy(this->data_, in.data_, (in.size())*sizeof(T));
            }
            else if(in.dimension() == 1)
                marray_detail::OperateHelperBinary<1, marray_detail::Assign<T, TLocal>, T, TLocal, isConstLocal, A, ALocal>::operate(*this, in, marray_detail::Assign<T, TLocal>(), this->data_, &in(0));
            else if(in.dimension() == 2)
                marray_detail::OperateHelperBinary<2, marray_detail::Assign<T, TLocal>, T, TLocal, isConstLocal, A, ALocal>::operate(*this, in, marray_detail::Assign<T, TLocal>(), this->data_, &in(0));
            else if(in.dimension() == 3)
                marray_detail::OperateHelperBinary<3, marray_detail::Assign<T, TLocal>, T, TLocal, isConstLocal, A, ALocal>::operate(*this, in, marray_detail::Assign<T, TLocal>(), this->data_, &in(0));
            else if(in.dimension() == 4)
                marray_detail::OperateHelperBinary<4, marray_detail::Assign<T, TLocal>, T, TLocal, isConstLocal, A, ALocal>::operate(*this, in, marray_detail::Assign<T, TLocal>(), this->data_, &in(0));
            else if(in.dimension() == 5)
                marray_detail::OperateHelperBinary<5, marray_detail::Assign<T, TLocal>, T, TLocal, isConstLocal, A, ALocal>::operate(*this, in, marray_detail::Assign<T, TLocal>(), this->data_, &in(0));
            else if(in.dimension() == 6)
                marray_detail::OperateHelperBinary<6, marray_detail::Assign<T, TLocal>, T, TLocal, isConstLocal, A, ALocal>::operate(*this, in, marray_detail::Assign<T, TLocal>(), this->data_, &in(0));
            else if(in.dimension() == 7)
                marray_detail::OperateHelperBinary<7, marray_detail::Assign<T, TLocal>, T, TLocal, isConstLocal, A, ALocal>::operate(*this, in, marray_detail::Assign<T, TLocal>(), this->data_, &in(0));
            else if(in.dimension() == 8)
                marray_detail::OperateHelperBinary<8, marray_detail::Assign<T, TLocal>, T, TLocal, isConstLocal, A, ALocal>::operate(*this, in, marray_detail::Assign<T, TLocal>(), this->data_, &in(0));
            else if(in.dimension() == 9)
                marray_detail::OperateHelperBinary<9, marray_detail::Assign<T, TLocal>, T, TLocal, isConstLocal, A, ALocal>::operate(*this, in, marray_detail::Assign<T, TLocal>(), this->data_, &in(0));
            else if(in.dimension() == 10)
                marray_detail::OperateHelperBinary<10, marray_detail::Assign<T, TLocal>, T, TLocal, isConstLocal, A, ALocal>::operate(*this, in, marray_detail::Assign<T, TLocal>(), this->data_, &in(0));
            else {
                typename View<TLocal, isConstLocal, ALocal>::const_iterator it = in.begin();
                for(std::size_t j=0; j<this->size(); ++j, ++it) {
                    this->data_[j] = static_cast<T>(*it);
                }
            }
        }
    }
    testInvariant();
    return *this;
}

/// Assignment.
///
/// \param value Value.
///
/// All entries are set to value.
///
template<class T, class A> 
inline Marray<T, A>& 
Marray<T, A>::operator=
(
    const T& value
)
{
    marray_detail::Assert(MARRAY_NO_DEBUG || this->data_ != 0);
    for(std::size_t j=0; j<this->size(); ++j) {
        this->data_[j] = value;
    }
    return *this;
}

template<class T, class A> 
template<class E, class Te>
inline Marray<T, A>&
Marray<T, A>::operator=
(
    const ViewExpression<E, Te>& expression
)
{
    if(expression.overlaps(*this)) {
        Marray<T, A> m(expression); // temporary copy
        (*this) = m; // recursive call
    }
    else {
        // re-allocate memory (if necessary)
        if(this->size() != expression.size()) {
            dataAllocator_.deallocate(this->data_, this->size());
            this->data_ = dataAllocator_.allocate(expression.size());
        }
        
        // copy geometry
        this->geometry_.resize(expression.dimension());
        for(std::size_t j=0; j<expression.dimension(); ++j) {
            this->geometry_.shape(j) = expression.shape(j);
        }
        this->geometry_.size() = expression.size();
        this->geometry_.isSimple() = true;
        this->geometry_.coordinateOrder() = expression.coordinateOrder();
        if(this->geometry_.dimension() != 0) {
            marray_detail::stridesFromShape(this->geometry_.shapeBegin(), this->geometry_.shapeEnd(),
                this->geometry_.shapeStridesBegin(), this->geometry_.coordinateOrder());
            marray_detail::stridesFromShape(this->geometry_.shapeBegin(), this->geometry_.shapeEnd(),
                this->geometry_.stridesBegin(), this->geometry_.coordinateOrder());
        }
        
        // copy data
        marray_detail::operate(*this, expression, marray_detail::Assign<T, Te>());
    }
    return *this;
}

template<class T, class A> 
template<bool SKIP_INITIALIZATION, class ShapeIterator>
inline void
Marray<T, A>::resizeHelper
(
    ShapeIterator begin,
    ShapeIterator end,
    const T& value
)
{   
    testInvariant();
    // compute size
    std::vector<std::size_t> newShape;
    std::size_t newSize = 1;
    for(ShapeIterator it = begin; it != end; ++it) {
        std::size_t x = static_cast<std::size_t>(*it);
        marray_detail::Assert(MARRAY_NO_ARG_TEST || x > 0);
        newShape.push_back(x);
        newSize *= x;
    }
    // allocate new
    value_type* newData = dataAllocator_.allocate(newSize); 
    if(!SKIP_INITIALIZATION) {
        for(std::size_t j=0; j<newSize; ++j) {
            newData[j] = value;
        }
    }
    // copy old data in region of overlap
    if(this->data_ != 0) {
        if(newSize == 1 || this->dimension() == 0) {
            newData[0] = this->data_[0];
        }
        else {
            std::vector<std::size_t> base1(this->dimension());
            std::vector<std::size_t> base2(newShape.size());
            std::vector<std::size_t> shape1(this->dimension(), 1);
            std::vector<std::size_t> shape2(newShape.size(), 1);
            for(std::size_t j=0; j<std::min(this->dimension(), newShape.size()); ++j) {
                shape1[j] = std::min(this->shape(j), newShape[j]);
                shape2[j] = shape1[j];
            }
            View<T, true, A> view1;
            this->constView(base1.begin(), shape1.begin(), view1);
            View<T, false, A> viewT(newShape.begin(), newShape.end(),
                newData, this->coordinateOrder(),
                this->coordinateOrder());
            View<T, false, A> view2;
            viewT.view(base2.begin(), shape2.begin(), view2);
            view1.squeeze();
            view2.squeeze();
            view2 = view1; // copy
        }
        dataAllocator_.deallocate(this->data_, this->size()); 
        this->data_ = 0;
    }
    base::assign(begin, end, newData, this->geometry_.coordinateOrder(),
        this->geometry_.coordinateOrder());
    testInvariant();
}

/// Resize (existing entries are preserved, new entries are initialized).
///
/// \param begin Iterator to the beginning of a sequence that determines
/// the new shape.
/// \param end Iterator to the end of that sequence.
/// \param value Initial value to be assigned to newly allocated entries.
///
template<class T, class A> 
template<class ShapeIterator>
void
Marray<T, A>::resize
(
    ShapeIterator begin,
    ShapeIterator end,
    const T& value
)
{   
    resizeHelper<false>(begin, end, value);
}

/// Resize (existing entries are preserved).
///
/// \param is Flag to be set to SkipInitialization.
/// \param begin Iterator to the beginning of a sequence that determines
/// the new shape.
/// \param end Iterator to the end of that sequence.
///
template<class T, class A> 
template<class ShapeIterator>
void
Marray<T, A>::resize
(
    const InitializationSkipping& is,
    ShapeIterator begin,
    ShapeIterator end
)
{   
    resizeHelper<true>(begin, end);
}

#ifdef HAVE_CPP11_INITIALIZER_LISTS
/// Resize (existing entries are preserved, new entries are initialized).
///
/// \param shape Shape given as initializer list.
/// \param value Initial value to be assigned to newly allocated entries.
///
template<class T, class A> 
inline void
Marray<T, A>::resize
(
    std::initializer_list<std::size_t> shape,
    const T& value
)
{
    resizeHelper<false>(shape.begin(), shape.end(), value);
}

/// Resize (existing entries are preserved).
///
/// \param shape Shape given as initializer list.
/// \param value Initial value to be assigned to newly allocated entries.
///
template<class T, class A> 
inline void
Marray<T, A>::resize
(
    const InitializationSkipping& si,
    std::initializer_list<std::size_t> shape
)
{
    resizeHelper<true>(shape.begin(), shape.end());
}
#endif

/// Invariant test.
///
template<class T, class A> 
inline void
Marray<T, A>::testInvariant() const
{
    View<T, false, A>::testInvariant();
    marray_detail::Assert(MARRAY_NO_DEBUG || this->geometry_.isSimple());
}

// iterator implementation

/// Invariant test.
///
template<class T, bool isConst, class A>
inline void 
Iterator<T, isConst, A>::testInvariant() const
{
    if(!MARRAY_NO_DEBUG) {
        if(view_ == 0) { 
            marray_detail::Assert(coordinates_.size() == 0 
                && index_ == 0
                && pointer_ == 0);
        }
        else { 
            if(view_->size() == 0) { // un-initialized view
                marray_detail::Assert(coordinates_.size() == 0 
                    && index_ == 0
                    && pointer_ == 0);
            }
            else { // initialized view
                marray_detail::Assert(index_ >= 0 && index_ <= view_->size());
                if(index_ == view_->size()) { // end iterator
                    marray_detail::Assert(pointer_ == &((*view_)(view_->size()-1)) + 1);
                }
                else {
                    marray_detail::Assert(pointer_ == &((*view_)(index_)));
                }
                if(!view_->isSimple()) {
                    marray_detail::Assert(coordinates_.size() == view_->dimension());
                    if(index_ == view_->size()) { // end iterator
                        if(view_->coordinateOrder() == LastMajorOrder) {
                            marray_detail::Assert(coordinates_[0] == view_->shape(0));
                            for(std::size_t j=1; j<coordinates_.size(); ++j) {
                                marray_detail::Assert(coordinates_[j] == view_->shape(j)-1);
                            }
                        }
                        else { // FirstMajorOrder
                            std::size_t d = view_->dimension() - 1;
                            marray_detail::Assert(coordinates_[d] == view_->shape(d));
                            for(std::size_t j=0; j<d; ++j) {
                                marray_detail::Assert(coordinates_[j] == view_->shape(j)-1);
                            }
                        }
                    }
                    else {
                        std::vector<std::size_t> testCoord(coordinates_.size());
                        view_->indexToCoordinates(index_, testCoord.begin());
                        for(std::size_t j=0; j<coordinates_.size(); ++j) {
                            marray_detail::Assert(coordinates_[j] == testCoord[j]);
                        }
                    }
                }
            }
        }
    }
}

/// Empty constructor.
template<class T, bool isConst, class A>
inline Iterator<T, isConst, A>::Iterator()
:   view_(0),
    pointer_(0),
    index_(0),
    coordinates_(std::vector<std::size_t>())
{
    testInvariant();
}

/// Construct from View on constant data.
///
/// \param view View
/// \param index Index into the View.
///
template<class T, bool isConst, class A>
inline Iterator<T, isConst, A>::Iterator
(
    const View<T, true, A>& view,
    const std::size_t index
)
:   view_(&view),
    pointer_(0),
    index_(index),
    coordinates_(std::vector<std::size_t>(view.dimension()))
    // Note for developers: If isConst==false, the construction view_(&view)
    // fails due to incompatible types. This is intended because it should 
    // not be possible to construct a mutable iterator on constant data.
{
    if(view.size() == 0) { // un-initialized view
        marray_detail::Assert(MARRAY_NO_ARG_TEST || index == 0);
    }
    else {
        if(view.isSimple()) {
            marray_detail::Assert(MARRAY_NO_ARG_TEST || index <= view.size());
            pointer_ = &view(0) + index;
        }
        else {
            if(index >= view.size()) { // end iterator
                if(view_->coordinateOrder() == LastMajorOrder) {
                    coordinates_[0] = view.shape(0);
                    for(std::size_t j=1; j<view.dimension(); ++j) {
                        coordinates_[j] = view.shape(j)-1;
                    }
                }
                else { // FirstMajorOrder
                    std::size_t d = view_->dimension() - 1;
                    coordinates_[d] = view.shape(d);
                    for(std::size_t j=0; j<d; ++j) {
                        coordinates_[j] = view.shape(j)-1;
                    }
                }
                pointer_ = &view(view.size()-1) + 1;
            }
            else {
                view.indexToCoordinates(index, coordinates_.begin());
                pointer_ = &view(index);
            }
        }
    }
    testInvariant();
}

/// Construct from View on mutable data.
///
/// \param view View
/// \param index Index into the View.
///
template<class T, bool isConst, class A>
inline Iterator<T, isConst, A>::Iterator
(
    const View<T, false, A>& view,
    const std::size_t index
)
:   view_(reinterpret_cast<view_pointer>(&view)),
    pointer_(0),
    index_(index),
    coordinates_(std::vector<std::size_t>(view.dimension()))
    // Note for developers: If isConst==true, the construction
    // view_(reinterpret_cast<view_pointer>(&view)) works as well.
    // This is intended because it should be possible to construct 
    // a constant iterator on mutable data.
{
    if(view.size() == 0) { // un-initialized view
        marray_detail::Assert(MARRAY_NO_ARG_TEST || index == 0);
    }
    else {
        if(view.isSimple()) {
            marray_detail::Assert(MARRAY_NO_ARG_TEST || index <= view.size());
            pointer_ = &view(0) + index;
        }
        else {
            if(index >= view.size()) { // end iterator
                if(view_->coordinateOrder() == LastMajorOrder) {
                    coordinates_[0] = view.shape(0);
                    for(std::size_t j=1; j<view.dimension(); ++j) {
                        coordinates_[j] = view.shape(j)-1;
                    }
                }
                else { // FirstMajorOrder
                    std::size_t d = view_->dimension() - 1;
                    coordinates_[d] = view.shape(d);
                    for(std::size_t j=0; j<d; ++j) {
                        coordinates_[j] = view.shape(j)-1;
                    }
                }
                pointer_ = &view(view.size()-1) + 1;
            }
            else {
                view.indexToCoordinates(index, coordinates_.begin());
                pointer_ = &view(index);
            }
        }
    }
    testInvariant();
}

/// Construct from View on mutable data.
///
/// \param view View
/// \param index Index into the View.
///
template<class T, bool isConst, class A>
inline Iterator<T, isConst, A>::Iterator
(
    View<T, false, A>& view,
    const std::size_t index
)
:   view_(reinterpret_cast<view_pointer>(&view)),
    pointer_(0),
    index_(index),
    coordinates_(std::vector<std::size_t>(view.dimension()))
    // Note for developers: If isConst==true, the construction
    // view_(reinterpret_cast<view_pointer>(&view)) works as well.
    // This is intended because it should be possible to construct 
    // a constant iterator on mutable data.
{
    if(view.size() == 0) { // un-initialized view
        marray_detail::Assert(MARRAY_NO_ARG_TEST || index == 0);
    }
    else {
        if(view.isSimple()) {
            marray_detail::Assert(MARRAY_NO_ARG_TEST || index <= view.size());
            pointer_ = &view(0) + index;
        }
        else {
            if(index >= view.size()) { // end iterator
                if(view_->coordinateOrder() == LastMajorOrder) {
                    coordinates_[0] = view.shape(0);
                    for(std::size_t j=1; j<view.dimension(); ++j) {
                        coordinates_[j] = view.shape(j)-1;
                    }
                }
                else { // FirstMajorOrder
                    std::size_t d = view_->dimension() - 1;
                    coordinates_[d] = view.shape(d);
                    for(std::size_t j=0; j<d; ++j) {
                        coordinates_[j] = view.shape(j)-1;
                    }
                }
                pointer_ = &view(view.size()-1) + 1;
            }
            else {
                view.indexToCoordinates(index, coordinates_.begin());
                pointer_ = &view(index);
            }
        }
    }
    testInvariant();
}

/// Copy constructor or conversion from an Iterator on mutable data.
/// 
template<class T, bool isConst, class A>
inline Iterator<T, isConst, A>::Iterator
(
    const Iterator<T, false, A>& in
)
:   view_(view_pointer(in.view_)),
    pointer_(pointer(in.pointer_)), 
    index_(in.index_),
    coordinates_(in.coordinates_)  
{
    testInvariant();
}

/// De-reference.
///
template<class T, bool isConst, class A>
inline typename Iterator<T, isConst, A>::reference
Iterator<T, isConst, A>::operator*() const
{
    marray_detail::Assert(MARRAY_NO_DEBUG || (view_ != 0 && index_ < view_->size()));
    return *pointer_;
}

/// Pointer.
///
template<class T, bool isConst, class A>
inline typename Iterator<T, isConst, A>::pointer
Iterator<T, isConst, A>::operator->() const
{
    marray_detail::Assert(MARRAY_NO_DEBUG || (view_ != 0 && index_ < view_->size()));
    return pointer_;
}

/// Element access.
///
template<class T, bool isConst, class A>
inline typename Iterator<T, isConst, A>::reference
Iterator<T, isConst, A>::operator[]
(
    const std::size_t x
) const
{
    marray_detail::Assert(MARRAY_NO_DEBUG || (view_ != 0 && x+index_ < view_->size()));
    return (*view_)(x+index_);
}

template<class T, bool isConst, class A>
inline Iterator<T, isConst, A>&
Iterator<T, isConst, A>::operator+=
(
    const difference_type& x
)
{
    marray_detail::Assert(MARRAY_NO_DEBUG || view_ != 0);
    if(index_ < view_->size()) { // view initialized and iterator not at the end
        if(index_ + x < view_->size()) {
            index_ += x;
            if(view_->isSimple()) {
                pointer_ += x;
            }
            else {
                pointer_ = &((*view_)(index_));
                view_->indexToCoordinates(index_, coordinates_.begin());
            }
        }
        else {
            // set to end iterator
            index_ = view_->size();
            if(view_->isSimple()) {
                pointer_ = &(*view_)(0) + view_->size();
            }
            else {
                pointer_ = (&(*view_)(view_->size()-1)) + 1;
                view_->indexToCoordinates(view_->size()-1, coordinates_.begin());
                if(view_->coordinateOrder() == LastMajorOrder) {
                    ++coordinates_[0];
                }
                else { // FirstMajorOrder
                    ++coordinates_[view_->dimension()-1];
                }
            }
        }
    }
    testInvariant();
    return *this;
}

template<class T, bool isConst, class A>
inline Iterator<T, isConst, A>&
Iterator<T, isConst, A>::operator-=
(
    const difference_type& x
)
{
    marray_detail::Assert(MARRAY_NO_DEBUG || view_ != 0);
    marray_detail::Assert(MARRAY_NO_ARG_TEST || static_cast<difference_type>(index_) >= x);
    index_ -= x;
    if(view_->isSimple()) {
        pointer_ -= x;
    }
    else {
        pointer_ = &((*view_)(index_));
        view_->indexToCoordinates(index_, coordinates_.begin());
    }
    testInvariant();
    return *this;
}

/// Prefix increment.
///
template<class T, bool isConst, class A>
inline Iterator<T, isConst, A>&
Iterator<T, isConst, A>::operator++()
{
    marray_detail::Assert(MARRAY_NO_DEBUG || view_ != 0);
    if(index_ < view_->size()) { // view initialized and iterator not at the end
        ++index_;
        if(view_->isSimple()) {
            ++pointer_;
        }
        else {
            if(index_ < view_->size()) {
                if(view_->coordinateOrder() == LastMajorOrder) {
                    for(std::size_t j=0; j<coordinates_.size(); ++j) {
                        if(coordinates_[j] == view_->shape(j)-1) {
                            pointer_ -= view_->strides(j) * coordinates_[j];
                            coordinates_[j] = 0;
                        }
                        else {
                            pointer_ += view_->strides(j);
                            ++coordinates_[j];
                            break;
                        }
                    }
                }
                else { // FirstMajorOrder
                    std::size_t j = coordinates_.size() - 1;
                    for(;;) {
                        if(coordinates_[j] == view_->shape(j)-1) {
                            pointer_ -= view_->strides(j) * coordinates_[j];
                            coordinates_[j] = 0;
                        }
                        else {
                            pointer_ += view_->strides(j);
                            ++coordinates_[j];
                            break;
                        }
                        if(j == 0) {
                            break;
                        }
                        else {
                            --j;
                        }
                    }
                }
            }
            else {
                // set to end iterator
                pointer_ = &((*view_)(view_->size()-1)) + 1;
                if(view_->coordinateOrder() == LastMajorOrder) {
                    ++coordinates_[0];
                }
                else { // FirstMajorOrder
                    ++coordinates_[view_->dimension()-1];
                }
            }
        }
    }
    testInvariant();
    return *this;
}

/// Prefix decrement.
///
template<class T, bool isConst, class A>
inline Iterator<T, isConst, A>&
Iterator<T, isConst, A>::operator--()
{
    marray_detail::Assert(MARRAY_NO_DEBUG || (view_ != 0 && index_ > 0));
    --index_;
    if(view_->isSimple()) {
        --pointer_;
    }
    else {
        if(index_ == view_->size()) { 
            // decrement from end iterator
            --pointer_;
            if(view_->coordinateOrder() == LastMajorOrder) {
                --coordinates_[0];
            }
            else { // FirstMajorOrder
                --coordinates_[view_->dimension() - 1];
            }
        }
        else {
            if(view_->coordinateOrder() == LastMajorOrder) {
                for(std::size_t j=0; j<coordinates_.size(); ++j) {
                    if(coordinates_[j] == 0) {
                        coordinates_[j] = view_->shape(j)-1;
                        pointer_ += view_->strides(j) * coordinates_[j];
                    }
                    else {
                        pointer_ -= view_->strides(j);
                        --coordinates_[j];
                        break;
                    }
                }
            }
            else { // FirstMajorOrder
                std::size_t j = view_->dimension() - 1;
                for(;;) {
                    if(coordinates_[j] == 0) {
                        coordinates_[j] = view_->shape(j)-1;
                        pointer_ += view_->strides(j) * coordinates_[j];
                    }
                    else {
                        pointer_ -= view_->strides(j);
                        --coordinates_[j];
                        break;
                    }
                    if(j == 0) {
                        break;
                    }
                    else {
                        --j;
                    }
                }
            }
        }
    }
    testInvariant();
    return *this;
}

/// Postfix increment.
///
template<class T, bool isConst, class A>
inline Iterator<T, isConst, A> 
Iterator<T, isConst, A>::operator++(int)
{
    marray_detail::Assert(MARRAY_NO_DEBUG || view_ != 0);
    Iterator<T, isConst, A> copy = *this;
    ++(*this);
    return copy;
}

/// Postfix decrement.
///
template<class T, bool isConst, class A>
inline Iterator<T, isConst, A> 
Iterator<T, isConst, A>::operator--(int)
{
    marray_detail::Assert(MARRAY_NO_DEBUG || (view_ != 0 && index_ > 0));
    Iterator<T, isConst, A> copy = *this;
    --(*this);
    return copy;
}

template<class T, bool isConst, class A>
inline Iterator<T, isConst, A>
Iterator<T, isConst, A>::operator+
(
    const difference_type& x
) const
{
    marray_detail::Assert(MARRAY_NO_DEBUG || view_ != 0);
    Iterator<T, isConst, A> tmp = *this;
    tmp += x;
    return tmp;
}

template<class T, bool isConst, class A>
inline Iterator<T, isConst, A>
Iterator<T, isConst, A>::operator-
(
    const difference_type& x
) const
{
    marray_detail::Assert(MARRAY_NO_DEBUG || view_ != 0);
    Iterator<T, isConst, A> tmp = *this;
    tmp -= x;
    return tmp;
}

template<class T, bool isConst, class A>
template<bool isConstLocal>
inline typename Iterator<T, isConst, A>::difference_type
Iterator<T, isConst, A>::operator-
(
    const Iterator<T, isConstLocal, A>& it
) const
{
    marray_detail::Assert(MARRAY_NO_DEBUG || view_ != 0);
    marray_detail::Assert(MARRAY_NO_ARG_TEST || it.view_ != 0);
    return difference_type(index_)-difference_type(it.index_);
}

template<class T, bool isConst, class A>
template<bool isConstLocal> 
inline bool
Iterator<T, isConst, A>::operator==
(
    const Iterator<T, isConstLocal, A>& it
) const
{
    marray_detail::Assert(MARRAY_NO_DEBUG || view_ != 0);
    marray_detail::Assert(MARRAY_NO_ARG_TEST || (it.view_ != 0 && (void*)it.view_ == (void*)view_));
    return index_ == it.index_;
}

template<class T, bool isConst, class A>
template<bool isConstLocal>
inline bool
Iterator<T, isConst, A>::operator!=
(
    const Iterator<T, isConstLocal, A>& it
) const
{
    marray_detail::Assert(MARRAY_NO_DEBUG || view_ != 0);
    marray_detail::Assert(MARRAY_NO_ARG_TEST || it.view_ != 0);
    marray_detail::Assert(MARRAY_NO_ARG_TEST || 
        static_cast<const void*>(it.view_) == static_cast<const void*>(view_));
    return index_ != it.index_;
}

template<class T, bool isConst, class A>
template<bool isConstLocal>
inline bool
Iterator<T, isConst, A>::operator<
(
    const Iterator<T, isConstLocal, A>& it
) const
{
    marray_detail::Assert(MARRAY_NO_DEBUG || view_ != 0);
    marray_detail::Assert(MARRAY_NO_ARG_TEST || (it.view_ != 0 && it.view_ == view_));
    return(index_ < it.index_);
}

template<class T, bool isConst, class A>
template<bool isConstLocal>
inline bool
Iterator<T, isConst, A>::operator>
(
    const Iterator<T, isConstLocal, A>& it
) const
{
    marray_detail::Assert(MARRAY_NO_DEBUG || view_ != 0);
    marray_detail::Assert(MARRAY_NO_ARG_TEST || (it.view_ != 0 && it.view_ == view_));
    return(index_ > it.index_);
}

template<class T, bool isConst, class A>
template<bool isConstLocal>
inline bool
Iterator<T, isConst, A>::operator<=
(
    const Iterator<T, isConstLocal, A>& it
) const
{
    marray_detail::Assert(MARRAY_NO_DEBUG || view_ != 0);
    marray_detail::Assert(MARRAY_NO_ARG_TEST || (it.view_ != 0 && it.view_ == view_));
    return(index_ <= it.index_);
}

template<class T, bool isConst, class A>
template<bool isConstLocal> 
inline bool
Iterator<T, isConst, A>::operator>=
(
    const Iterator<T, isConstLocal, A>& it
) const
{
    marray_detail::Assert(MARRAY_NO_DEBUG || view_ != 0);
    marray_detail::Assert(MARRAY_NO_ARG_TEST || (it.view_ != 0 && it.view_ == view_));
    return(index_ >= it.index_);
}

/// Fast alternative to comparing with the end iterator.
///
/// \return Boolean indicator.
///
template<class T, bool isConst, class A>
inline bool
Iterator<T, isConst, A>::hasMore() const
{
    marray_detail::Assert(MARRAY_NO_DEBUG || view_ != 0);
    return index_ < view_->size();
}

/// Get the corresponding index in the View.
///
/// \return index Index.
///
template<class T, bool isConst, class A>
inline std::size_t 
Iterator<T, isConst, A>::index() const
{
    return index_;
}

/// Get the corresponding coordinate sequence in the View.
///
/// \param it Iterator into a container starting from which the
/// coordinate sequence is to be written (output).
///
template<class T, bool isConst, class A>
template<class CoordinateIterator>
inline void
Iterator<T, isConst, A>::coordinate
(
    CoordinateIterator it
) const
{
    marray_detail::Assert(MARRAY_NO_DEBUG || view_ != 0);
    marray_detail::Assert(MARRAY_NO_ARG_TEST || index_ < view_->size());
    if(view_->isSimple()) {
        view_->indexToCoordinates(index_, it);
    }
    else {
        for(std::size_t j=0; j<coordinates_.size(); ++j, ++it) {
            *it = coordinates_[j];
        }
    }
}

// implementation of expression templates

/// Expression template for efficient arithmetic operations.
template<class E, class T>
class ViewExpression {
public:
    typedef E expression_type;
    typedef T value_type;

    const std::size_t dimension() const 
        { return static_cast<const E&>(*this).dimension(); }
    const std::size_t size() const 
        { return static_cast<const E&>(*this).size(); }
    const std::size_t shape(const std::size_t j) const 
        { return static_cast<const E&>(*this).shape(j); }
    const std::size_t* shapeBegin() const 
        { return static_cast<const E&>(*this).shapeBegin(); }
    const std::size_t* shapeEnd() const 
        { return static_cast<const E&>(*this).shapeEnd(); }
    template<class Tv, bool isConst, class A> 
        bool overlaps(const View<Tv, isConst, A>& v) const
            { return static_cast<const E&>(*this).overlaps(v); }
    const CoordinateOrder& coordinateOrder() const 
        { return static_cast<const E&>(*this).coordinateOrder(); }
    const bool isSimple() const
        { return static_cast<const E&>(*this).isSimple(); }
    template<class Accessor>
        const T& operator()(Accessor it) const
            { return static_cast<const E&>(*this)(it); }
    const T& operator()(const std::size_t c0, const std::size_t c1) const
        { return static_cast<const E&>(*this)(c0, c1); }
    const T& operator()(const std::size_t c0, const std::size_t c1, const std::size_t c2) const 
        { return static_cast<const E&>(*this)(c0, c1, c2); }
    const T& operator()(const std::size_t c0, const std::size_t c1, const std::size_t c2, const std::size_t c3) const 
        { return static_cast<const E&>(*this)(c0, c1, c2, c3); }
    const T& operator()(const std::size_t c0, const std::size_t c1, const std::size_t c2, const std::size_t c3, const std::size_t c4) const 
        { return static_cast<const E&>(*this)(c0, c1, c2, c3, c4); }
    const T& operator[](const std::size_t offset) const
        { return static_cast<const E&>(*this)[offset]; }
    operator E&() 
        { return static_cast<E&>(*this); }
    operator E const&() const 
        { return static_cast<const E&>(*this); }

    // \cond suppress_doxygen
    class ExpressionIterator {
    public:
        ExpressionIterator(const ViewExpression<E, T>& expression)
        : expression_(expression), // cast!
          data_(&expression_(0)),
          offset_(0)
            {}
        void incrementCoordinate(const std::size_t coordinateIndex)
            { offset_ += expression_.strides(coordinateIndex); }
        void resetCoordinate(const std::size_t coordinateIndex)
            { offset_ -= expression_.strides(coordinateIndex)

                         * (expression_.shape(coordinateIndex) - 1); }
        const T& operator*() const
            { // return expression_[offset_]; 
              // would require making this nested class a friend of View
              // which in turn would require a forward declaration of 
              // this class. work around:
              return data_[offset_]; }
    private:
        const E& expression_;
        const T* data_;
        std::size_t offset_;
    };
    // \endcond suppress_doxygen
};

// \cond suppress_doxygen
template<class E, class T, class UnaryFunctor>
class UnaryViewExpression 
: public ViewExpression<UnaryViewExpression<E, T, UnaryFunctor>, T> 
{
public:
    typedef E expression_type;
    typedef T value_type;

    UnaryViewExpression(const ViewExpression<E, T>& e)
        : e_(e), // cast!
          unaryFunctor_(UnaryFunctor()) 
        {}
    const std::size_t dimension() const 
        { return e_.dimension(); }
    const std::size_t size() const 
        { return e_.size(); }
    const std::size_t shape(const std::size_t j) const 
        { return e_.shape(j); }
    const std::size_t* shapeBegin() const 
        { return e_.shapeBegin(); }
    const std::size_t* shapeEnd() const 
        { return e_.shapeEnd(); }
    template<class Tv, bool isConst, class A> 
        bool overlaps(const View<Tv, isConst, A>& v) const
            { return e_.overlaps(v); }
    const CoordinateOrder& coordinateOrder() const 
        { return e_.coordinateOrder(); }
    const bool isSimple() const
        { return e_.isSimple(); }
    template<class Accessor>
        const T operator()(Accessor it) const
            { return unaryFunctor_(e_(it)); }
    const T operator()(const std::size_t c0, const std::size_t c1) const
        { return unaryFunctor_(e_(c0, c1)); }
    const T operator()(const std::size_t c0, const std::size_t c1,const std::size_t c2) const 
        { return unaryFunctor_(e_(c0, c1, c2)); }
    const T operator()(const std::size_t c0, const std::size_t c1,const std::size_t c2, const std::size_t c3) const 
        { return unaryFunctor_(e_(c0, c1, c2, c3)); }
    const T operator()(const std::size_t c0, const std::size_t c1,const std::size_t c2, const std::size_t c3, const std::size_t c4) const 
        { return unaryFunctor_(e_(c0, c1, c2, c3, c4)); }
    const T operator[](const std::size_t offset) const
        { return unaryFunctor_(e_[offset]); }

    class ExpressionIterator {
    public:
        ExpressionIterator(const UnaryViewExpression<E, T, UnaryFunctor>& expression)
        : unaryFunctor_(expression.unaryFunctor_),
          iterator_(expression.e_)
            {}
        void incrementCoordinate(const std::size_t coordinateIndex)
            { iterator_.incrementCoordinate(coordinateIndex); } 
        void resetCoordinate(const std::size_t coordinateIndex)
            { iterator_.resetCoordinate(coordinateIndex); }
        const T operator*() const
            { return unaryFunctor_(*iterator_); }
    private:
        UnaryFunctor unaryFunctor_;
        typename E::ExpressionIterator iterator_;
    };

private:
    const E& e_;
    UnaryFunctor unaryFunctor_;
};

template<class E1, class T1, class E2, class T2, class BinaryFunctor>
class BinaryViewExpression
: public ViewExpression<BinaryViewExpression<E1, T1, E2, T2, BinaryFunctor>, 
                        typename marray_detail::PromoteType<T1, T2>::type>
{
public:
    typedef E1 expression_type_1;
    typedef E2 expression_type_2;
    typedef T1 value_type_1;
    typedef T2 value_type_2;
    typedef typename marray_detail::PromoteType<T1, T2>::type value_type;
    typedef BinaryFunctor functor_type;
    typedef ViewExpression<BinaryViewExpression<E1, T1, E2, T2, BinaryFunctor>, 
        value_type> base;

    BinaryViewExpression(const ViewExpression<E1, T1>& e1, 
        const ViewExpression<E2, T2>& e2) 
        : e1_(e1), e2_(e2), // cast!
          binaryFunctor_(BinaryFunctor()) 
        {
            if(!MARRAY_NO_DEBUG) {
                marray_detail::Assert(e1_.size() != 0 && e2_.size() != 0);
                marray_detail::Assert(e1_.dimension() == e2_.dimension());
                for(std::size_t j=0; j<e1_.dimension(); ++j) {
                    marray_detail::Assert(e1_.shape(j) == e2_.shape(j));
                }
            }
        }
    const std::size_t dimension() const 
        { return e1_.dimension() < e2_.dimension() ? e2_.dimension() : e1_.dimension(); }
    const std::size_t size() const 
        { return e1_.size() < e2_.size() ? e2_.size() : e1_.size(); }
    const std::size_t shape(const std::size_t j) const 
        { return e1_.dimension() < e2_.dimension() ? e2_.shape(j) : e1_.shape(j); }
    const std::size_t* shapeBegin() const 
        { return e1_.dimension() < e2_.dimension() ? e2_.shapeBegin() : e1_.shapeBegin(); }
    const std::size_t* shapeEnd() const 
        { return e1_.dimension() < e2_.dimension() ? e2_.shapeEnd() : e1_.shapeEnd(); }
    template<class Tv, bool isConst, class A> 
        bool overlaps(const View<Tv, isConst, A>& v) const
            { return e1_.overlaps(v) || e2_.overlaps(v); }
    const CoordinateOrder& coordinateOrder() const 
        { return e1_.coordinateOrder(); }
    const bool isSimple() const
        { return e1_.isSimple() && e2_.isSimple() 
                 && e1_.coordinateOrder() == e2_.coordinateOrder(); }
    template<class Accessor>
        const value_type operator()(Accessor it) const
            { return binaryFunctor_(e1_(it), e2_(it)); }
    const value_type operator()(const std::size_t c0, const std::size_t c1) const
        { return binaryFunctor_(e1_(c0, c1), e2_(c0, c1)); }
    const value_type operator()(const std::size_t c0, const std::size_t c1, const std::size_t c2) const 
        { return binaryFunctor_(e1_(c0, c1, c2), e2_(c0, c1, c2)); }
    const value_type operator()(const std::size_t c0, const std::size_t c1, const std::size_t c2, const std::size_t c3) const 
        { return binaryFunctor_(e1_(c0, c1, c2, c3), e2_(c0, c1, c2, c3)); }
    const value_type operator()(const std::size_t c0, const std::size_t c1, const std::size_t c2, const std::size_t c3, const std::size_t c4) const 
        { return binaryFunctor_(e1_(c0, c1, c2, c3, c4), e2_(c0, c1, c2, c3, c4)); }
    const value_type operator[](const std::size_t offset) const
        { return binaryFunctor_(e1_[offset], e2_[offset]); }

    class ExpressionIterator {
    public:
        ExpressionIterator(const BinaryViewExpression<E1, T1, E2, T2, BinaryFunctor>& expression)
        : binaryFunctor_(expression.binaryFunctor_),
          iterator1_(expression.e1_), 
          iterator2_(expression.e2_)
            {}
        void incrementCoordinate(const std::size_t coordinateIndex)
            {   iterator1_.incrementCoordinate(coordinateIndex); 
                iterator2_.incrementCoordinate(coordinateIndex); }
        void resetCoordinate(const std::size_t coordinateIndex)
            {   iterator1_.resetCoordinate(coordinateIndex); 
                iterator2_.resetCoordinate(coordinateIndex); }
        const value_type operator*() const
            { return binaryFunctor_(*iterator1_, *iterator2_); }
    private:
        BinaryFunctor binaryFunctor_;
        typename E1::ExpressionIterator iterator1_;
        typename E2::ExpressionIterator iterator2_;
    };

private:
    const expression_type_1& e1_;
    const expression_type_2& e2_;
    BinaryFunctor binaryFunctor_;
};

template<class E, class T, class S, class BinaryFunctor>
class BinaryViewExpressionScalarFirst
: public ViewExpression<BinaryViewExpressionScalarFirst<E, T, S, BinaryFunctor>, 
                        typename marray_detail::PromoteType<T, S>::type> {
public:
    typedef E expression_type;
    typedef T value_type_1;
    typedef S scalar_type;
    typedef typename marray_detail::PromoteType<T, S>::type value_type;
    typedef BinaryFunctor functor_type;
    typedef ViewExpression<BinaryViewExpressionScalarFirst<E, T, S, BinaryFunctor>, 
        value_type> base;

    BinaryViewExpressionScalarFirst(const ViewExpression<E, T>& e, 
        const scalar_type& scalar) 
        : e_(e), // cast!
          scalar_(scalar), binaryFunctor_(BinaryFunctor()) 
        { }
    const std::size_t dimension() const 
        { return e_.dimension(); }
    const std::size_t size() const 
        { return e_.size(); }
    const std::size_t shape(const std::size_t j) const 
        { return e_.shape(j); }
    const std::size_t* shapeBegin() const 
        { return e_.shapeBegin(); }
    const std::size_t* shapeEnd() const 
        { return e_.shapeEnd(); }
    template<class Tv, bool isConst, class A> 
        bool overlaps(const View<Tv, isConst, A>& v) const
            { return e_.overlaps(v); }
    const CoordinateOrder& coordinateOrder() const 
        { return e_.coordinateOrder(); }
    const bool isSimple() const
        { return e_.isSimple(); }
    template<class Accessor>
        const value_type operator()(Accessor it) const
            { return binaryFunctor_(scalar_, e_(it)); }
    const value_type operator()(const std::size_t c0, const std::size_t c1) const
        { return binaryFunctor_(scalar_, e_(c0, c1)); }
    const value_type operator()(const std::size_t c0, const std::size_t c1, const std::size_t c2) const 
        { return binaryFunctor_(scalar_, e_(c0, c1, c2)); }
    const value_type operator()(const std::size_t c0, const std::size_t c1, const std::size_t c2, const std::size_t c3) const 
        { return binaryFunctor_(scalar_, e_(c0, c1, c2, c3)); }
    const value_type operator()(const std::size_t c0, const std::size_t c1, const std::size_t c2, const std::size_t c3, const std::size_t c4) const 
        { return binaryFunctor_(scalar_, e_(c0, c1, c2, c3, c4)); }
    const value_type operator[](const std::size_t offset) const
        { return binaryFunctor_(scalar_, e_[offset]); }

    class ExpressionIterator {
    public:
        ExpressionIterator(const BinaryViewExpressionScalarFirst<E, T, S, BinaryFunctor>& expression)
        : binaryFunctor_(expression.binaryFunctor_),
          scalar_(expression.scalar_),
          iterator_(expression.e_)
            {}
        void incrementCoordinate(const std::size_t coordinateIndex)
            { iterator_.incrementCoordinate(coordinateIndex); }
        void resetCoordinate(const std::size_t coordinateIndex)
            { iterator_.resetCoordinate(coordinateIndex); }
        const T operator*() const
            { return binaryFunctor_(scalar_, *iterator_); }
    private:
        BinaryFunctor binaryFunctor_;
        const typename BinaryViewExpressionScalarFirst<E, T, S, BinaryFunctor>::scalar_type& scalar_;
        typename E::ExpressionIterator iterator_;
    };

private:
    const expression_type& e_;
    const scalar_type scalar_;
    BinaryFunctor binaryFunctor_;
};

template<class E, class T, class S, class BinaryFunctor>
class BinaryViewExpressionScalarSecond
: public ViewExpression<BinaryViewExpressionScalarSecond<E, T, S, BinaryFunctor>, 
                        typename marray_detail::PromoteType<T, S>::type> {
public:
    typedef T value_type_1;
    typedef E expression_type;
    typedef S scalar_type;
    typedef typename marray_detail::PromoteType<T, S>::type value_type;
    typedef BinaryFunctor functor_type;
    typedef ViewExpression<BinaryViewExpressionScalarSecond<E, T, S, BinaryFunctor>, 
        value_type> base;

    BinaryViewExpressionScalarSecond(const ViewExpression<E, T>& e, 
        const scalar_type& scalar) 
        : e_(e), // cast!
          scalar_(scalar), binaryFunctor_(BinaryFunctor())
        { }
    const std::size_t dimension() const 
        { return e_.dimension(); }
    const std::size_t size() const 
        { return e_.size(); }
    const std::size_t shape(const std::size_t j) const 
        { return e_.shape(j); }
    const std::size_t* shapeBegin() const 
        { return e_.shapeBegin(); }
    const std::size_t* shapeEnd() const 
        { return e_.shapeEnd(); }
    template<class Tv, bool isConst, class A> 
        bool overlaps(const View<Tv, isConst, A>& v) const
            { return e_.overlaps(v); }
    const CoordinateOrder& coordinateOrder() const 
        { return e_.coordinateOrder(); }
    const bool isSimple() const
        { return e_.isSimple(); }
    template<class Accessor>
        const value_type operator()(Accessor it) const
            { return binaryFunctor_(e_(it), scalar_); }
    const value_type operator()(const std::size_t c0, const std::size_t c1) const
        { return binaryFunctor_(e_(c0, c1), scalar_); }
    const value_type operator()(const std::size_t c0, const std::size_t c1, const std::size_t c2) const 
        { return binaryFunctor_(e_(c0, c1, c2), scalar_); }
    const value_type operator()(const std::size_t c0, const std::size_t c1, const std::size_t c2, const std::size_t c3) const 
        { return binaryFunctor_(e_(c0, c1, c2, c3), scalar_); }
    const value_type operator()(const std::size_t c0, const std::size_t c1, const std::size_t c2, const std::size_t c3, const std::size_t c4) const 
        { return binaryFunctor_(e_(c0, c1, c2, c3, c4), scalar_); }
    const value_type operator[](const std::size_t offset) const
        { return binaryFunctor_(e_[offset], scalar_); }

    class ExpressionIterator {
    public:
        ExpressionIterator(const BinaryViewExpressionScalarSecond<E, T, S, BinaryFunctor>& expression)
        : binaryFunctor_(expression.binaryFunctor_),
          scalar_(expression.scalar_),
          iterator_(expression.e_)
            {}
        void incrementCoordinate(const std::size_t coordinateIndex)
            { iterator_.incrementCoordinate(coordinateIndex); }
        void resetCoordinate(const std::size_t coordinateIndex)
            { iterator_.resetCoordinate(coordinateIndex); }
        const T operator*() const
            { return binaryFunctor_(*iterator_, scalar_); }
    private:
        BinaryFunctor binaryFunctor_;
        const typename BinaryViewExpressionScalarSecond<E, T, S, BinaryFunctor>::scalar_type& scalar_;
        typename E::ExpressionIterator iterator_;
    };

private:
    const expression_type& e_;
    const scalar_type scalar_;
    BinaryFunctor binaryFunctor_;
};
// \endcond suppress_doxygen

// implementation of marray_detail 

// \cond suppress_doxygen
namespace marray_detail { 

template<class A>
class Geometry 
{
public:
    typedef typename A::template rebind<std::size_t>::other allocator_type;

    Geometry(const allocator_type& = allocator_type());
    Geometry(const std::size_t, const CoordinateOrder& = defaultOrder,
        const std::size_t = 0, const bool = true, 
        const allocator_type& = allocator_type());
    template<class ShapeIterator>
        Geometry(ShapeIterator, ShapeIterator,
            const CoordinateOrder& = defaultOrder,
            const CoordinateOrder& = defaultOrder,
            const allocator_type& = allocator_type());
    template<class ShapeIterator, class StrideIterator>
        Geometry(ShapeIterator, ShapeIterator, StrideIterator,
            const CoordinateOrder& = defaultOrder, 
            const allocator_type& = allocator_type());
    Geometry(const Geometry<A>&);
    ~Geometry();

    Geometry<A>& operator=(const Geometry<A>&);

    void resize(const std::size_t dimension);
    const std::size_t dimension() const;
    const std::size_t shape(const std::size_t) const;
    std::size_t& shape(const std::size_t);
    const std::size_t shapeStrides(const std::size_t) const;
    std::size_t& shapeStrides(const std::size_t);
    const std::size_t strides(const std::size_t) const;
    std::size_t& strides(const std::size_t);
    const std::size_t* shapeBegin() const;
    std::size_t* shapeBegin();
    const std::size_t* shapeEnd() const;
    std::size_t* shapeEnd();
    const std::size_t* shapeStridesBegin() const;
    std::size_t* shapeStridesBegin();
    const std::size_t* shapeStridesEnd() const;
    std::size_t* shapeStridesEnd();
    const std::size_t* stridesBegin() const;
    std::size_t* stridesBegin();
    const std::size_t* stridesEnd() const;
    std::size_t* stridesEnd();
    const std::size_t size() const;
    std::size_t& size();
    const CoordinateOrder& coordinateOrder() const;
    CoordinateOrder& coordinateOrder();
    const bool isSimple() const;
    void updateSimplicity();
    bool& isSimple();

private:
    allocator_type allocator_;  
    std::size_t* shape_;
    std::size_t* shapeStrides_;
        // Intended redundancy: shapeStrides_ could be
        // computed from shape_ and coordinateOrder_
    std::size_t* strides_;
    std::size_t dimension_;
    std::size_t size_;
        // intended redundancy: size_ could be computed from shape_
    CoordinateOrder coordinateOrder_;
    bool isSimple_; 
        // simple array: an array which is unstrided (i.e. the strides
        // equal the shape strides), cf. the function testInvariant of 
        // View for the formal definition.
};

template<class A>
inline 
Geometry<A>::Geometry
(
    const typename Geometry<A>::allocator_type& allocator
) 
: allocator_(allocator),
  shape_(0), 
  shapeStrides_(0), 
  strides_(0), 
  dimension_(0),
  size_(0), 
  coordinateOrder_(defaultOrder), 
  isSimple_(true)
{
}

template<class A>
inline 
Geometry<A>::Geometry
(
    const Geometry<A>& g
)
: allocator_(g.allocator_),
  shape_(g.dimension_==0 ? 0 : allocator_.allocate(g.dimension_*3)), 
  shapeStrides_(shape_ + g.dimension_), 
  strides_(shapeStrides_ + g.dimension_), 
  dimension_(g.dimension_),
  size_(g.size_), 
  coordinateOrder_(g.coordinateOrder_), 
  isSimple_(g.isSimple_)
{
    /*
    for(std::size_t j=0; j<dimension_; ++j) {
        shape_[j] = g.shape_[j];
        shapeStrides_[j] = g.shapeStrides_[j];
        strides_[j] = g.strides_[j];
    }
    */
    memcpy(shape_, g.shape_, (dimension_*3)*sizeof(std::size_t));
}

template<class A>
inline 
Geometry<A>::Geometry
(
    const std::size_t dimension,
    const CoordinateOrder& order,
    const std::size_t size,
    const bool isSimple,
    const typename Geometry<A>::allocator_type& allocator
)
: allocator_(allocator),
  shape_(allocator_.allocate(dimension*3)), 
  shapeStrides_(shape_+dimension),
  strides_(shapeStrides_+dimension), 
  dimension_(dimension),
  size_(size),
  coordinateOrder_(order),
  isSimple_(isSimple)
{
}

template<class A>
template<class ShapeIterator>
inline 
Geometry<A>::Geometry
(
    ShapeIterator begin, 
    ShapeIterator end,
    const CoordinateOrder& externalCoordinateOrder,
    const CoordinateOrder& internalCoordinateOrder,
    const typename Geometry<A>::allocator_type& allocator
)
: allocator_(allocator),
  shape_(allocator_.allocate(std::distance(begin, end) * 3)), 
  shapeStrides_(shape_ + std::distance(begin, end)),
  strides_(shapeStrides_ + std::distance(begin, end)), 
  dimension_(std::distance(begin, end)),
  size_(1),
  coordinateOrder_(internalCoordinateOrder),
  isSimple_(true)
{
    if(dimension_ != 0) { // if the array is not a scalar
        isSimple_ = (externalCoordinateOrder == internalCoordinateOrder);
        for(std::size_t j=0; j<dimension(); ++j, ++begin) {
            const std::size_t s = static_cast<std::size_t>(*begin);
            shape(j) = s;
            size() *= s;
        }
        stridesFromShape(shapeBegin(), shapeEnd(), stridesBegin(), 
            externalCoordinateOrder);
        stridesFromShape(shapeBegin(), shapeEnd(), shapeStridesBegin(), 
            internalCoordinateOrder);
    }
}

template<class A>
template<class ShapeIterator, class StrideIterator>
inline
Geometry<A>::Geometry
(
    ShapeIterator begin, 
    ShapeIterator end,
    StrideIterator it,
    const CoordinateOrder& internalCoordinateOrder,
    const typename Geometry<A>::allocator_type& allocator
)
: allocator_(allocator),
  shape_(allocator_.allocate(std::distance(begin, end) * 3)), 
  shapeStrides_(shape_ + std::distance(begin, end)),
  strides_(shapeStrides_ + std::distance(begin, end)), 
  dimension_(std::distance(begin, end)),
  size_(1),
  coordinateOrder_(internalCoordinateOrder),
  isSimple_(true)
{
    if(dimension() != 0) {
        for(std::size_t j=0; j<dimension(); ++j, ++begin, ++it) {
            const std::size_t s = static_cast<std::size_t>(*begin);
            shape(j) = s;
            size() *= s;
            strides(j) = *it;
        }
        stridesFromShape(shapeBegin(), shapeEnd(), shapeStridesBegin(), 
            internalCoordinateOrder);
        updateSimplicity();
    }
}

template<class A>
inline 
Geometry<A>::~Geometry()
{
    allocator_.deallocate(shape_, dimension_*3); 
}

template<class A>
inline Geometry<A>& 
Geometry<A>::operator=
(
    const Geometry<A>& g
)
{
    if(&g != this) { // no self-assignment
        if(g.dimension_ != dimension_) {
            allocator_.deallocate(shape_, dimension_*3);
            dimension_ = g.dimension_;
            shape_ = allocator_.allocate(dimension_*3);
            shapeStrides_ = shape_+dimension_;
            strides_ = shapeStrides_+dimension_;
            dimension_ = g.dimension_;
        }
        /*
        for(std::size_t j=0; j<dimension_; ++j) {
            shape_[j] = g.shape_[j];
            shapeStrides_[j] = g.shapeStrides_[j];
            strides_[j] = g.strides_[j];
        }
        */
        memcpy(shape_, g.shape_, (dimension_*3)*sizeof(std::size_t));
        size_ = g.size_;
        coordinateOrder_ = g.coordinateOrder_;
        isSimple_ = g.isSimple_;
    }
    return *this;
}

template<class A>
inline void 
Geometry<A>::resize
(
    const std::size_t dimension
)
{
    if(dimension != dimension_) {
        std::size_t* newShape = allocator_.allocate(dimension*3);
        std::size_t* newShapeStrides = newShape + dimension;
        std::size_t* newStrides = newShapeStrides + dimension; 
        for(std::size_t j=0; j<( (dimension < dimension_) ? dimension : dimension_); ++j) {
            // save existing entries
            newShape[j] = shape(j);
            newShapeStrides[j] = shapeStrides(j);
            newStrides[j] = strides(j);
        }
        allocator_.deallocate(shape_, dimension_*3);
        shape_ = newShape;
        shapeStrides_ = newShapeStrides;
        strides_ = newStrides;
        dimension_ = dimension;
    }
}

template<class A>
inline const std::size_t 
Geometry<A>::dimension() const
{
    return dimension_; 
}

template<class A>
inline const std::size_t 
Geometry<A>::shape(const std::size_t j) const
{ 
    Assert(MARRAY_NO_DEBUG || j<dimension_); 
    return shape_[j]; 
}

template<class A>
inline std::size_t& 
Geometry<A>::shape(const std::size_t j)
{ 
    Assert(MARRAY_NO_DEBUG || j<dimension_); 
    return shape_[j]; 
}

template<class A>
inline const std::size_t 
Geometry<A>::shapeStrides
(
    const std::size_t j
) const
{ 
    Assert(MARRAY_NO_DEBUG || j<dimension_); 
    return shapeStrides_[j]; 
}

template<class A>
inline std::size_t& 
Geometry<A>::shapeStrides
(
    const std::size_t j
)
{ 
    Assert(MARRAY_NO_DEBUG || j<dimension_); 
    return shapeStrides_[j]; 
}

template<class A>
inline const std::size_t 
Geometry<A>::strides
(
    const std::size_t j
) const
{ 
    Assert(MARRAY_NO_DEBUG || j<dimension_); 
    return strides_[j]; 
}

template<class A>
inline std::size_t& 
Geometry<A>::strides
(
    const std::size_t j
)
{ 
    Assert(MARRAY_NO_DEBUG || j<dimension_); 
    return strides_[j]; 
}

template<class A>
inline const std::size_t* 
Geometry<A>::shapeBegin() const
{ 
    return shape_; 
}

template<class A>
inline std::size_t* 
Geometry<A>::shapeBegin()
{ 
    return shape_; 
}

template<class A>
inline const std::size_t* 
Geometry<A>::shapeEnd() const 
{ 
    return shape_ + dimension_; 
}

template<class A>
inline std::size_t* 
Geometry<A>::shapeEnd() 
{ 
    return shape_ + dimension_; 
}

template<class A>
inline const std::size_t* 
Geometry<A>::shapeStridesBegin() const 
{ 
    return shapeStrides_; 
}

template<class A>
inline std::size_t* 
Geometry<A>::shapeStridesBegin() 
{ 
    return shapeStrides_; 
}

template<class A>
inline const std::size_t* 
Geometry<A>::shapeStridesEnd() const 
{ 
    return shapeStrides_ + dimension_; 
}

template<class A>
inline std::size_t* 
Geometry<A>::shapeStridesEnd() 
{ 
    return shapeStrides_ + dimension_; 
}

template<class A>
inline const std::size_t* 
Geometry<A>::stridesBegin() const 
{ 
    return strides_; 
}

template<class A>
inline std::size_t* 
Geometry<A>::stridesBegin() 
{ 
    return strides_; 
}

template<class A>
inline const std::size_t* 
Geometry<A>::stridesEnd() const 
{ 
    return strides_ + dimension_; 
}

template<class A>
inline std::size_t* 
Geometry<A>::stridesEnd() 
{ 
    return strides_ + dimension_; 
}

template<class A>
inline const std::size_t 
Geometry<A>::size() const
{ 
    return size_; 
}

template<class A>
inline std::size_t& 
Geometry<A>::size()
{ 
    return size_; 
}

template<class A>
inline const CoordinateOrder& 
Geometry<A>::coordinateOrder() const
{ 
    return coordinateOrder_; 
}

template<class A>
inline CoordinateOrder& 
Geometry<A>::coordinateOrder()
{ 
    return coordinateOrder_; 
}

template<class A>
inline const bool 
Geometry<A>::isSimple() const 
{ 
    return isSimple_; 
}

template<class A>
inline bool& 
Geometry<A>::isSimple()
{ 
    return isSimple_; 
}

template<class A>
inline void
Geometry<A>::updateSimplicity()
{ 
    for(std::size_t j=0; j<dimension(); ++j) {
        if(shapeStrides(j) != strides(j)) {
            isSimple_ = false;
            return;
        }
    }
    isSimple_ = true; 
    // a 0-dimensional geometry is simple
}

template<class ShapeIterator, class StridesIterator>
inline void 
stridesFromShape
(
    ShapeIterator begin,
    ShapeIterator end,
    StridesIterator strideBegin,
    const CoordinateOrder& coordinateOrder
) 
{
    Assert(MARRAY_NO_ARG_TEST || std::distance(begin, end) != 0);
    std::size_t dimension = std::distance(begin, end);
    ShapeIterator shapeIt;
    StridesIterator strideIt;
    if(coordinateOrder == FirstMajorOrder) {
        shapeIt = begin + (dimension-1);
        strideIt = strideBegin + (dimension-1);
        *strideIt = 1;
        for(std::size_t j=1; j<dimension; ++j) {
            std::size_t tmp = *strideIt;
            --strideIt;
            (*strideIt) = tmp * (*shapeIt);
            --shapeIt;
        }
    }
    else {
        shapeIt = begin;
        strideIt = strideBegin;
        *strideIt = 1;
        for(std::size_t j=1; j<dimension; ++j) {
            std::size_t tmp = *strideIt;
            ++strideIt;
            (*strideIt) = tmp * (*shapeIt);
            ++shapeIt;
        }
    }
}

template<unsigned short N, class Functor, class T, class A>
struct OperateHelperUnary
{
    static inline void operate
    (
        View<T, false, A>& v, 
        Functor f, 
        T* data
    )
    {
        for(std::size_t j=0; j<v.shape(N-1); ++j) {
            OperateHelperUnary<N-1, Functor, T, A>::operate(v, f, data);
            data += v.strides(N-1);
        }
        data -= v.shape(N-1) * v.strides(N-1);
    }
};


template<class Functor, class T, class A>
struct OperateHelperUnary<0, Functor, T, A>
{
    static inline void operate
    (
        View<T, false, A>& v, 
        Functor f, 
        T* data
    )
    { 
        f(*data); 
    }
};

template<unsigned short N, class Functor, class T1, class T2, class A>
struct OperateHelperBinaryScalar
{
    static inline void operate
    (
        View<T1, false, A>& v, 
        const T2& x, 
        Functor f, 
        T1* data
    )
    {
        for(std::size_t j=0; j<v.shape(N-1); ++j) {
            OperateHelperBinaryScalar<N-1, Functor, T1, T2, A>::operate(
                v, x, f, data);
            data += v.strides(N-1);
        }
        data -= v.shape(N-1) * v.strides(N-1);
    }
};

template<class Functor, class T1, class T2, class A>
struct OperateHelperBinaryScalar<0, Functor, T1, T2, A>
{
    static inline void operate
    (
        View<T1, false, A>& v, 
        const T2& x, 
        Functor f, 
        T1* data
    )
    { 
        f(*data, x); 
    }
};

template<unsigned short N, class Functor, class T1, class T2, 
         bool isConst, class A1, class A2>
struct OperateHelperBinary
{
    static inline void operate
    (
        View<T1, false, A1>& v, 
        const View<T2, isConst, A2>& w, 
        Functor f, 
        T1* data1,
        const T2* data2
    )
    {
        for(std::size_t j=0; j<v.shape(N-1); ++j) {
            OperateHelperBinary<N-1, Functor, T1, T2, isConst, A1, A2>::operate(
                v, w, f, data1, data2);
            data1 += v.strides(N-1);
            data2 += w.strides(N-1);
        }
        data1 -= v.shape(N-1) * v.strides(N-1);
        data2 -= w.shape(N-1) * w.strides(N-1);
    }
};

template<class Functor, class T1, class T2, bool isConst, class A1, class A2>
struct OperateHelperBinary<0, Functor, T1, T2, isConst, A1, A2>
{
    static inline void operate
    (
        View<T1, false, A1>& v, 
        const View<T2, isConst, A2>& w, 
        Functor f, 
        T1* data1,
        const T2* data2
    )
    {
        f(*data1, *data2); 
    }
};

template<class TFrom, class TTo, class AFrom, class ATo> 
struct AssignmentOperatorHelper<false, TFrom, TTo, AFrom, ATo>
{
    // from constant to mutable 
    //
    // here, 'to' must be initialized (which is asserted) because
    // otherwise, the pointer to.data_ to mutable data would have 
    // to be initialized with the pointer from.data_ to constant 
    // data which we don't do.
    //
    static void execute
    (
        const View<TFrom, true, AFrom>& from,
        View<TTo, false, ATo>& to
    )
    {
        typedef typename View<TFrom, true, AFrom>::const_iterator FromIterator;
        typedef typename View<TTo, false, ATo>::iterator ToIterator;
        if(!MARRAY_NO_ARG_TEST) {
            Assert(from.data_ != 0 && from.dimension() == to.dimension());
            for(std::size_t j=0; j<from.dimension(); ++j) {
                Assert(from.shape(j) == to.shape(j));
            }
        }
        if(from.overlaps(to)) {
            Marray<TFrom, AFrom> m = from; // temporary copy
            execute(m, to);
        }
        else if(from.coordinateOrder() == to.coordinateOrder() 
                && from.isSimple() && to.isSimple()
                && IsEqual<TFrom, TTo>::type) {
            memcpy(to.data_, from.data_, (from.size())*sizeof(TFrom));
        }
        else if(from.dimension() == 1)
            OperateHelperBinary<1, Assign<TTo, TFrom>, TTo, TFrom, true, ATo, AFrom>::operate(to, from, Assign<TTo, TFrom>(), &to(0), &from(0));
        else if(from.dimension() == 2)
            OperateHelperBinary<2, Assign<TTo, TFrom>, TTo, TFrom, true, ATo, AFrom>::operate(to, from, Assign<TTo, TFrom>(), &to(0), &from(0));
        else if(from.dimension() == 3)
            OperateHelperBinary<3, Assign<TTo, TFrom>, TTo, TFrom, true, ATo, AFrom>::operate(to, from, Assign<TTo, TFrom>(), &to(0), &from(0));
        else if(from.dimension() == 4)
            OperateHelperBinary<4, Assign<TTo, TFrom>, TTo, TFrom, true, ATo, AFrom>::operate(to, from, Assign<TTo, TFrom>(), &to(0), &from(0));
        else if(from.dimension() == 5)
            OperateHelperBinary<5, Assign<TTo, TFrom>, TTo, TFrom, true, ATo, AFrom>::operate(to, from, Assign<TTo, TFrom>(), &to(0), &from(0));
        else if(from.dimension() == 6)
            OperateHelperBinary<6, Assign<TTo, TFrom>, TTo, TFrom, true, ATo, AFrom>::operate(to, from, Assign<TTo, TFrom>(), &to(0), &from(0));
        else if(from.dimension() == 7)
            OperateHelperBinary<7, Assign<TTo, TFrom>, TTo, TFrom, true, ATo, AFrom>::operate(to, from, Assign<TTo, TFrom>(), &to(0), &from(0));
        else if(from.dimension() == 8)
            OperateHelperBinary<8, Assign<TTo, TFrom>, TTo, TFrom, true, ATo, AFrom>::operate(to, from, Assign<TTo, TFrom>(), &to(0), &from(0));
        else if(from.dimension() == 9)
            OperateHelperBinary<9, Assign<TTo, TFrom>, TTo, TFrom, true, ATo, AFrom>::operate(to, from, Assign<TTo, TFrom>(), &to(0), &from(0));
        else if(from.dimension() == 10)
            OperateHelperBinary<10, Assign<TTo, TFrom>, TTo, TFrom, true, ATo, AFrom>::operate(to, from, Assign<TTo, TFrom>(), &to(0), &from(0));
        else {
            FromIterator itFrom = from.begin();
            ToIterator itTo = to.begin();
            for(; itFrom.hasMore(); ++itFrom, ++itTo) {
                *itTo = static_cast<TTo>(*itFrom);
            }
        }
    }

    /// from mutable to mutable.
    ///
    /// here, 'to' need not be initialized.
    ///
    static void execute
    (
        const View<TFrom, false, AFrom>& from,
        View<TTo, false, ATo>& to
    )
    {
        typedef typename View<TFrom, false, AFrom>::const_iterator FromIterator;
        typedef typename View<TTo, false, ATo>::iterator ToIterator;
        if(static_cast<const void*>(&from) != static_cast<const void*>(&to)) { // no self-assignment
            if(to.data_ == 0) { // if the view 'to' is not initialized
                // initialize the view 'to' with source data
                Assert(MARRAY_NO_ARG_TEST || sizeof(TTo) == sizeof(TFrom));
                to.data_ = static_cast<TTo*>(static_cast<void*>(from.data_)); // copy pointer
                to.geometry_ = from.geometry_;
            }
            else { // if the view 'to' is initialized
                if(!MARRAY_NO_ARG_TEST) {
                    Assert(from.data_ != 0 && from.dimension() == to.dimension());
                    for(std::size_t j=0; j<from.dimension(); ++j) {
                        Assert(from.shape(j) == to.shape(j));
                    }
                }
                if(from.overlaps(to)) {
                    Marray<TFrom, AFrom> m = from; // temporary copy
                    execute(m, to);
                }
                else if(from.coordinateOrder() == to.coordinateOrder() 
                        && from.isSimple() && to.isSimple()
                        && IsEqual<TFrom, TTo>::type) {
                    memcpy(to.data_, from.data_, (from.size())*sizeof(TFrom));
                }
                else if(from.dimension() == 1)
                    OperateHelperBinary<1, Assign<TTo, TFrom>, TTo, TFrom, true, ATo, AFrom>::operate(to, from, Assign<TTo, TFrom>(), &to(0), &from(0));
                else if(from.dimension() == 2)
                    OperateHelperBinary<2, Assign<TTo, TFrom>, TTo, TFrom, true, ATo, AFrom>::operate(to, from, Assign<TTo, TFrom>(), &to(0), &from(0));
                else if(from.dimension() == 3)
                    OperateHelperBinary<3, Assign<TTo, TFrom>, TTo, TFrom, true, ATo, AFrom>::operate(to, from, Assign<TTo, TFrom>(), &to(0), &from(0));
                else if(from.dimension() == 4)
                    OperateHelperBinary<4, Assign<TTo, TFrom>, TTo, TFrom, true, ATo, AFrom>::operate(to, from, Assign<TTo, TFrom>(), &to(0), &from(0));
                else if(from.dimension() == 5)
                    OperateHelperBinary<5, Assign<TTo, TFrom>, TTo, TFrom, true, ATo, AFrom>::operate(to, from, Assign<TTo, TFrom>(), &to(0), &from(0));
                else if(from.dimension() == 6)
                    OperateHelperBinary<6, Assign<TTo, TFrom>, TTo, TFrom, true, ATo, AFrom>::operate(to, from, Assign<TTo, TFrom>(), &to(0), &from(0));
                else if(from.dimension() == 7)
                    OperateHelperBinary<7, Assign<TTo, TFrom>, TTo, TFrom, true, ATo, AFrom>::operate(to, from, Assign<TTo, TFrom>(), &to(0), &from(0));
                else if(from.dimension() == 8)
                    OperateHelperBinary<8, Assign<TTo, TFrom>, TTo, TFrom, true, ATo, AFrom>::operate(to, from, Assign<TTo, TFrom>(), &to(0), &from(0));
                else if(from.dimension() == 9)
                    OperateHelperBinary<9, Assign<TTo, TFrom>, TTo, TFrom, true, ATo, AFrom>::operate(to, from, Assign<TTo, TFrom>(), &to(0), &from(0));
                else if(from.dimension() == 10)
                    OperateHelperBinary<10, Assign<TTo, TFrom>, TTo, TFrom, true, ATo, AFrom>::operate(to, from, Assign<TTo, TFrom>(), &to(0), &from(0));
                else {
                    FromIterator itFrom = from.begin();
                    ToIterator itTo = to.begin();
                    for(; itFrom.hasMore(); ++itFrom, ++itTo) {
                        *itTo = static_cast<TTo>(*itFrom);
                    }
                }
            }
        }
    }
};

template<class TFrom, class TTo, class AFrom, class ATo> 
struct AssignmentOperatorHelper<true, TFrom, TTo, AFrom, ATo>
{
    /// from either constant or mutable to constant
    template<bool isConstFrom>
    static void execute
    (
        const View<TFrom, isConstFrom, AFrom>& from,
        View<TTo, true, ATo>& to
    )
    {
        Assert(MARRAY_NO_ARG_TEST || sizeof(TFrom) == sizeof(TTo));
        to.data_ = static_cast<const TTo*>(
            static_cast<const void*>(from.data_)); // copy pointer
        to.geometry_ = from.geometry_;
    }
};

template<>
struct AccessOperatorHelper<true>
{
    // access by scalar index
    template<class T, class U, bool isConst, class A>
    static typename View<T, isConst, A>::reference
    execute(const View<T, isConst, A>& v, const U& index)
    {
        v.testInvariant();
        Assert(MARRAY_NO_DEBUG || v.data_ != 0);
        Assert(MARRAY_NO_DEBUG || v.dimension() != 0 || index == 0);
        std::size_t offset;
        v.indexToOffset(index, offset);
        return v.data_[offset];
    }
};

template<>
struct AccessOperatorHelper<false>
{
    // access by iterator
    template<class T, class U, bool isConst, class A>
    static typename View<T, isConst, A>::reference
    execute(const View<T, isConst, A>& v, U it)
    {
        v.testInvariant();
        Assert(MARRAY_NO_DEBUG || v.data_ != 0);
        Assert(MARRAY_NO_DEBUG || v.dimension() != 0 || *it == 0);
        std::size_t offset;
        v.coordinatesToOffset(it, offset);
        return v.data_[offset];
    }
};

template<class Functor, class T, class A>
inline void 
operate
(
    View<T, false, A>& v, 
    Functor f
)
{
    if(v.isSimple()) {
        T* data = &v(0);
        for(std::size_t j=0; j<v.size(); ++j) {
            f(data[j]);
        }
    }
    else if(v.dimension() == 1)
        OperateHelperUnary<1, Functor, T, A>::operate(v, f, &v(0));
    else if(v.dimension() == 2)
        OperateHelperUnary<2, Functor, T, A>::operate(v, f, &v(0));
    else if(v.dimension() == 3)
        OperateHelperUnary<3, Functor, T, A>::operate(v, f, &v(0));
    else if(v.dimension() == 4)
        OperateHelperUnary<4, Functor, T, A>::operate(v, f, &v(0));
    else if(v.dimension() == 5)
        OperateHelperUnary<5, Functor, T, A>::operate(v, f, &v(0));
    else if(v.dimension() == 6)
        OperateHelperUnary<6, Functor, T, A>::operate(v, f, &v(0));
    else if(v.dimension() == 7)
        OperateHelperUnary<7, Functor, T, A>::operate(v, f, &v(0));
    else if(v.dimension() == 8)
        OperateHelperUnary<8, Functor, T, A>::operate(v, f, &v(0));
    else if(v.dimension() == 9)
        OperateHelperUnary<9, Functor, T, A>::operate(v, f, &v(0));
    else if(v.dimension() == 10)
        OperateHelperUnary<10, Functor, T, A>::operate(v, f, &v(0));
    else {
        for(typename View<T, false, A>::iterator it = v.begin(); it.hasMore(); ++it) {
            f(*it);
        }
    }
}

template<class Functor, class T, class A>
inline void 
operate
(
    View<T, false, A>& v, 
    const T& x, 
    Functor f
)
{
    if(v.isSimple()) {
        T* data = &v(0);
        for(std::size_t j=0; j<v.size(); ++j) {
            f(data[j], x);
        }
    }
    else if(v.dimension() == 1)
        OperateHelperBinaryScalar<1, Functor, T, T, A>::operate(v, x, f, &v(0));
    else if(v.dimension() == 2)
        OperateHelperBinaryScalar<2, Functor, T, T, A>::operate(v, x, f, &v(0));
    else if(v.dimension() == 3)
        OperateHelperBinaryScalar<3, Functor, T, T, A>::operate(v, x, f, &v(0));
    else if(v.dimension() == 4)
        OperateHelperBinaryScalar<4, Functor, T, T, A>::operate(v, x, f, &v(0));
    else if(v.dimension() == 5)
        OperateHelperBinaryScalar<5, Functor, T, T, A>::operate(v, x, f, &v(0));
    else if(v.dimension() == 6)
        OperateHelperBinaryScalar<6, Functor, T, T, A>::operate(v, x, f, &v(0));
    else if(v.dimension() == 7)
        OperateHelperBinaryScalar<7, Functor, T, T, A>::operate(v, x, f, &v(0));
    else if(v.dimension() == 8)
        OperateHelperBinaryScalar<8, Functor, T, T, A>::operate(v, x, f, &v(0));
    else if(v.dimension() == 9)
        OperateHelperBinaryScalar<9, Functor, T, T, A>::operate(v, x, f, &v(0));
    else if(v.dimension() == 10)
        OperateHelperBinaryScalar<10, Functor, T, T, A>::operate(v, x, f, &v(0));
    else {
        for(typename View<T, false, A>::iterator it = v.begin(); it.hasMore(); ++it) {
            f(*it, x); 
        }
    }
}

template<class Functor, class T1, class T2, bool isConst, class A>
inline void 
operate
(
    View<T1, false, A>& v, 
    const View<T2, isConst, A>& w, 
    Functor f
)
{
    if(!MARRAY_NO_ARG_TEST) {
        Assert(v.size() != 0 && w.size() != 0);
        Assert(w.dimension() == 0 || v.dimension() == w.dimension());
        if(w.dimension() != 0) {
            for(std::size_t j=0; j<v.dimension(); ++j) {
                Assert(v.shape(j) == w.shape(j));
            }
        }
    }
    if(w.dimension() == 0) {
        T2 x = w(0);
        if(v.isSimple()) {
            T1* dataV = &v(0);
            for(std::size_t j=0; j<v.size(); ++j) {
                f(dataV[j], x);
            }
        }
        else if(v.dimension() == 1)
            OperateHelperBinaryScalar<1, Functor, T1, T2, A>::operate(v, x, f, &v(0));
        else if(v.dimension() == 2)
            OperateHelperBinaryScalar<2, Functor, T1, T2, A>::operate(v, x, f, &v(0));
        else if(v.dimension() == 3)
            OperateHelperBinaryScalar<3, Functor, T1, T2, A>::operate(v, x, f, &v(0));
        else if(v.dimension() == 4)
            OperateHelperBinaryScalar<4, Functor, T1, T2, A>::operate(v, x, f, &v(0));
        else if(v.dimension() == 5)
            OperateHelperBinaryScalar<5, Functor, T1, T2, A>::operate(v, x, f, &v(0));
        else if(v.dimension() == 6)
            OperateHelperBinaryScalar<6, Functor, T1, T2, A>::operate(v, x, f, &v(0));
        else if(v.dimension() == 7)
            OperateHelperBinaryScalar<7, Functor, T1, T2, A>::operate(v, x, f, &v(0));
        else if(v.dimension() == 8)
            OperateHelperBinaryScalar<8, Functor, T1, T2, A>::operate(v, x, f, &v(0));
        else if(v.dimension() == 9)
            OperateHelperBinaryScalar<9, Functor, T1, T2, A>::operate(v, x, f, &v(0));
        else if(v.dimension() == 10)
            OperateHelperBinaryScalar<10, Functor, T1, T2, A>::operate(v, x, f, &v(0));
        else {
            for(typename View<T1, false>::iterator it = v.begin(); it.hasMore(); ++it) {
                f(*it, x);
            }
        }
    }
    else {
        if(v.overlaps(w)) {
            Marray<T2> m = w; // temporary copy
            operate(v, m, f); // recursive call
        }
        else {
            if(v.coordinateOrder() == w.coordinateOrder() 
                && v.isSimple() && w.isSimple()) {
                T1* dataV = &v(0);
                const T2* dataW = &w(0);
                for(std::size_t j=0; j<v.size(); ++j) {
                    f(dataV[j], dataW[j]);
                }
            }
            else if(v.dimension() == 1)
                OperateHelperBinary<1, Functor, T1, T2, isConst, A, A>::operate(v, w, f, &v(0), &w(0));
            else if(v.dimension() == 2)
                OperateHelperBinary<2, Functor, T1, T2, isConst, A, A>::operate(v, w, f, &v(0), &w(0));
            else if(v.dimension() == 3)
                OperateHelperBinary<3, Functor, T1, T2, isConst, A, A>::operate(v, w, f, &v(0), &w(0));
            else if(v.dimension() == 4)
                OperateHelperBinary<4, Functor, T1, T2, isConst, A, A>::operate(v, w, f, &v(0), &w(0));
            else if(v.dimension() == 5)
                OperateHelperBinary<5, Functor, T1, T2, isConst, A, A>::operate(v, w, f, &v(0), &w(0));
            else if(v.dimension() == 6)
                OperateHelperBinary<6, Functor, T1, T2, isConst, A, A>::operate(v, w, f, &v(0), &w(0));
            else if(v.dimension() == 7)
                OperateHelperBinary<7, Functor, T1, T2, isConst, A, A>::operate(v, w, f, &v(0), &w(0));
            else if(v.dimension() == 8)
                OperateHelperBinary<8, Functor, T1, T2, isConst, A, A>::operate(v, w, f, &v(0), &w(0));
            else if(v.dimension() == 9)
                OperateHelperBinary<9, Functor, T1, T2, isConst, A, A>::operate(v, w, f, &v(0), &w(0));
            else if(v.dimension() == 10)
                OperateHelperBinary<10, Functor, T1, T2, isConst, A, A>::operate(v, w, f, &v(0), &w(0));
            else {
                typename View<T1, false>::iterator itV = v.begin();
                typename View<T2, isConst>::const_iterator itW = w.begin();
                for(; itV.hasMore(); ++itV, ++itW) {
                    Assert(MARRAY_NO_DEBUG || itW.hasMore());
                    f(*itV, *itW);
                }
                Assert(MARRAY_NO_DEBUG || !itW.hasMore());
            }
        }
    }
}

template<class Functor, class T1, class A, class E, class T2>
inline void operate
(
    View<T1, false, A>& v, 
    const ViewExpression<E, T2>& expression, 
    Functor f
)
{
    const E& e = expression; // cast
    if(!MARRAY_NO_DEBUG) {
        Assert(v.size() != 0 && e.size() != 0);
        Assert(e.dimension() == v.dimension());
        if(v.dimension() == 0) {
            Assert(v.size() == 1 && e.size() == 1);
        }
        else {
            for(std::size_t j=0; j<v.dimension(); ++j) {
                Assert(v.shape(j) == e.shape(j));
            }
        }
    }
    if(e.overlaps(v)) {
        Marray<T1, A> m(e); // temporary copy
        operate(v, m, f);
    }
    else if(v.dimension() == 0) {
        f(v[0], e[0]);
    }
    else if(v.isSimple() && e.isSimple() 
    && v.coordinateOrder() == e.coordinateOrder()) {
        for(std::size_t j=0; j<v.size(); ++j) {
            f(v[j], e[j]);
        }
    }
    else {
        // loop unrolling does not improve performance here
        typename E::ExpressionIterator itE(e);
        std::size_t offsetV = 0;
        std::vector<std::size_t> coordinate(v.dimension());
        std::size_t maxDimension = v.dimension() - 1;
        for(;;) {
            f(v[offsetV], *itE);
            for(std::size_t j=0; j<v.dimension(); ++j) {
                if(coordinate[j]+1 == v.shape(j)) {
                    if(j == maxDimension) {
                        return;
                    }
                    else {
                        offsetV -= coordinate[j] * v.strides(j);
                        itE.resetCoordinate(j);
                        coordinate[j] = 0;
                    }
                }
                else {
                    offsetV += v.strides(j);
                    itE.incrementCoordinate(j);
                    ++coordinate[j];
                    break;
                }
            }
        }
    }
}

} // namespace marray_detail
// \endcond suppress_doxygen

} // namespace andres

#endif // #ifndef ANDRES_MARRAY_HXX
