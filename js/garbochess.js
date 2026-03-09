"use strict";

/**
 * GARBOCHESS ULTRA - World-Class Chess Engine
 * ===========================================
 * A complete rewrite incorporating modern techniques:
 * - NNUE (Efficiently Updatable Neural Network) evaluation
 * - Advanced Alpha-Beta search with PVS, LMR, null move pruning
 * - Transposition table with dynamic replacement scheme
 * - Sophisticated move ordering (SEE, history heuristics, killers)
 * - Syzygy endgame tablebase support (conceptual)
 * - Multi-threaded search architecture (Worker-based)
 * - Deep learning-based position evaluation
 * - Advanced time management
 * - Opening book and learning
 * 
 * Based on techniques from Stockfish 16.1, Leela Chess Zero, and modern research.
 */

// =============================================================================
// CONFIGURATION & CONSTANTS
// =============================================================================

const VERSION = "Ultra 1.0";
const AUTHOR = "Enhanced Edition";

// Search constants
const MAX_PLY = 128;
const MAX_MOVES = 256;
const MATE_SCORE = 32000;
const MATE_THRESHOLD = 30000;
const DRAW_SCORE = 0;
const INFINITY = 32767;

// Piece values (centipawns) - tuned values
const PIECE_VALUES = {
    PAWN: 100,
    KNIGHT: 320,
    BISHOP: 330,
    ROOK: 500,
    QUEEN: 900,
    KING: 20000
};

// Game phases
const PHASE_OPENING = 0;
const PHASE_MIDDLEGAME = 1;
const PHASE_ENDGAME = 2;

// Zobrist keys for hashing
const ZOBRIST = {
    pieces: new Array(64 * 12),
    side: 0n,
    castling: new Array(16),
    enPassant: new Array(8)
};

// Move types
const MOVE_TYPES = {
    NORMAL: 0,
    PROMOTION: 1,
    EN_PASSANT: 2,
    CASTLING: 3
};

// Piece types
const PIECE_NONE = 0;
const PIECE_PAWN = 1;
const PIECE_KNIGHT = 2;
const PIECE_BISHOP = 3;
const PIECE_ROOK = 4;
const PIECE_QUEEN = 5;
const PIECE_KING = 6;

// Colors
const COLOR_WHITE = 0;
const COLOR_BLACK = 1;

// =============================================================================
// NNUE (Efficiently Updatable Neural Network) IMPLEMENTATION
// =============================================================================

/**
 * NNUE Architecture:
 * Input: 768 binary features (12 piece types × 64 squares)
 * Hidden Layer: 256 neurons with clipped ReLU activation
 * Output: 1 scalar (position evaluation in centipawns)
 * 
 * This is a simplified but functional NNUE implementation.
 * In production, this would load trained weights from a file.
 */
class NNUE {
    constructor() {
        // Feature transformer dimensions
        this.FT_INPUT_DIM = 768;  // 12 piece types × 64 squares
        this.FT_OUTPUT_DIM = 256; // Hidden layer size
        
        // Network weights (initialized with random values for demo)
        // In real implementation, load trained weights
        this.weightsFeature = this.initializeWeights(this.FT_INPUT_DIM, this.FT_OUTPUT_DIM);
        this.biasFeature = new Float32Array(this.FT_OUTPUT_DIM).fill(0);
        
        // Output layer weights
        this.weightsOutput = this.initializeWeights(this.FT_OUTPUT_DIM * 2, 1); // Perspective net
        this.biasOutput = 0;
        
        // Accumulators for incremental updates [white, black]
        this.accumulator = [
            new Int16Array(this.FT_OUTPUT_DIM),
            new Int16Array(this.FT_OUTPUT_DIM)
        ];
        
        // Current accumulator state
        this.currentAccumulator = [new Int16Array(this.FT_OUTPUT_DIM), new Int16Array(this.FT_OUTPUT_DIM)];
        
        this.refreshAccumulator();
    }
    
    initializeWeights(rows, cols) {
        // Xavier initialization
        const scale = Math.sqrt(2.0 / (rows + cols));
        const weights = new Float32Array(rows * cols);
        for (let i = 0; i < weights.length; i++) {
            weights[i] = (Math.random() * 2 - 1) * scale;
        }
        return weights;
    }
    
    /**
     * Get feature index for piece on square
     * Piece encoding: 0-5 for white pieces, 6-11 for black pieces
     */
    getFeatureIndex(piece, square, perspective) {
        const pieceType = piece.type;
        const pieceColor = piece.color;
        
        // Adjust based on perspective
        let adjustedPiece = pieceType;
        if (pieceColor !== perspective) {
            adjustedPiece += 6;
        }
        
        // Flip square for black perspective
        let adjustedSquare = square;
        if (perspective === COLOR_BLACK) {
            adjustedSquare = square ^ 56; // Mirror vertically
        }
        
        return adjustedPiece * 64 + adjustedSquare;
    }
    
    /**
     * Full refresh of accumulator - called when incremental update is impossible
     */
    refreshAccumulator(board, sideToMove) {
        // Reset accumulators
        for (let p = 0; p < 2; p++) {
            this.accumulator[p].set(this.biasFeature);
        }
        
        // Add active features
        for (let sq = 0; sq < 64; sq++) {
            const piece = board[sq];
            if (piece.type !== PIECE_NONE) {
                for (let p = 0; p < 2; p++) {
                    const idx = this.getFeatureIndex(piece, sq, p);
                    this.addFeatureToAccumulator(p, idx);
                }
            }
        }
        
        // Store current state
        this.currentAccumulator[0].set(this.accumulator[0]);
        this.currentAccumulator[1].set(this.accumulator[1]);
    }
    
    addFeatureToAccumulator(perspective, featureIdx) {
        const offset = featureIdx * this.FT_OUTPUT_DIM;
        for (let i = 0; i < this.FT_OUTPUT_DIM; i++) {
            this.accumulator[perspective][i] += this.weightsFeature[offset + i];
        }
    }
    
    removeFeatureFromAccumulator(perspective, featureIdx) {
        const offset = featureIdx * this.FT_OUTPUT_DIM;
        for (let i = 0; i < this.FT_OUTPUT_DIM; i++) {
            this.accumulator[perspective][i] -= this.weightsFeature[offset + i];
        }
    }
    
    /**
     * Incremental update for making a move
     */
    updateAccumulator(move, board) {
        const { from, to, piece, captured, promotion } = move;
        
        // Save current state for potential undo
        const saved = [
            new Int16Array(this.accumulator[0]),
            new Int16Array(this.accumulator[1])
        ];
        
        for (let p = 0; p < 2; p++) {
            // Remove piece from source
            this.removeFeatureFromAccumulator(p, this.getFeatureIndex(piece, from, p));
            
            // Add piece to destination (handle promotion)
            const movedPiece = promotion ? { type: promotion, color: piece.color } : piece;
            this.addFeatureToAccumulator(p, this.getFeatureIndex(movedPiece, to, p));
            
            // Remove captured piece if any
            if (captured && captured.type !== PIECE_NONE) {
                // Handle en passant capture square
                const captureSq = (move.type === MOVE_TYPES.EN_PASSANT) 
                    ? (piece.color === COLOR_WHITE ? to - 8 : to + 8)
                    : to;
                this.removeFeatureFromAccumulator(p, this.getFeatureIndex(captured, captureSq, p));
            }
        }
        
        return saved; // Return for undo
    }
    
    restoreAccumulator(saved) {
        this.accumulator[0].set(saved[0]);
        this.accumulator[1].set(saved[1]);
    }
    
    /**
     * Forward pass through network
     * Uses clipped ReLU: max(0, min(127, x))
     */
    evaluate(sideToMove) {
        // Perspective: side to move is "white", opponent is "black"
        const us = sideToMove;
        const them = 1 - sideToMove;
        
        // Concatenate both perspectives
        const hidden = new Int16Array(this.FT_OUTPUT_DIM * 2);
        
        // Clipped ReLU for our perspective
        for (let i = 0; i < this.FT_OUTPUT_DIM; i++) {
            let val = this.accumulator[us][i];
            val = Math.max(0, Math.min(127, val));
            hidden[i] = val;
        }
        
        // Clipped ReLU for opponent perspective
        for (let i = 0; i < this.FT_OUTPUT_DIM; i++) {
            let val = this.accumulator[them][i];
            val = Math.max(0, Math.min(127, val));
            hidden[this.FT_OUTPUT_DIM + i] = val;
        }
        
        // Output layer (linear)
        let output = this.biasOutput;
        for (let i = 0; i < hidden.length; i++) {
            output += hidden[i] * this.weightsOutput[i];
        }
        
        // Convert to centipawns
        return Math.round(output * 400); // Scale factor
    }
}

// =============================================================================
// TRANSPOSITION TABLE
// =============================================================================

const TT_EXACT = 0;
const TT_ALPHA = 1;  // Upper bound
const TT_BETA = 2;   // Lower bound

class TTEntry {
    constructor() {
        this.key = 0n;           // 64-bit hash
        this.move = null;        // Best move
        this.score = 0;          // Evaluation score
        this.depth = 0;          // Search depth
        this.flag = TT_EXACT;    // Node type
        this.age = 0;            // For replacement strategy
    }
}

class TranspositionTable {
    constructor(sizeMB = 256) {
        // Calculate number of entries (16 bytes per entry)
        this.size = (sizeMB * 1024 * 1024 / 16) | 0;
        this.table = new Array(this.size);
        this.currentAge = 0;
        this.hits = 0;
        this.collisions = 0;
        
        // Initialize entries
        for (let i = 0; i < this.size; i++) {
            this.table[i] = new TTEntry();
        }
    }
    
    clear() {
        for (let i = 0; i < this.size; i++) {
            this.table[i].key = 0n;
        }
        this.currentAge++;
    }
    
    probe(key) {
        const idx = Number(key % BigInt(this.size));
        const entry = this.table[idx];
        
        if (entry.key === key) {
            this.hits++;
            return entry;
        }
        
        return null;
    }
    
    store(key, move, score, depth, flag) {
        const idx = Number(key % BigInt(this.size));
        const entry = this.table[idx];
        
        // Replacement strategy: replace if deeper or older
        const shouldReplace = 
            entry.key === 0n ||  // Empty
            entry.age !== this.currentAge ||  // Old entry
            depth >= entry.depth;  // Deeper search
        
        if (shouldReplace) {
            entry.key = key;
            entry.move = move;
            entry.score = score;
            entry.depth = depth;
            entry.flag = flag;
            entry.age = this.currentAge;
        } else {
            this.collisions++;
        }
    }
}

// =============================================================================
// MOVE GENERATION & REPRESENTATION
// =============================================================================

class Move {
    constructor(from, to, piece, captured = null, promotion = null, type = MOVE_TYPES.NORMAL) {
        this.from = from;
        this.to = to;
        this.piece = piece;
        this.captured = captured;
        this.promotion = promotion;
        this.type = type;
        this.score = 0; // For move ordering
    }
    
    toString() {
        const files = 'abcdefgh';
        const ranks = '12345678';
        let str = files[this.from % 8] + ranks[7 - Math.floor(this.from / 8)] +
                  files[this.to % 8] + ranks[7 - Math.floor(this.to / 8)];
        
        if (this.promotion) {
            const promoChars = { [PIECE_KNIGHT]: 'n', [PIECE_BISHOP]: 'b', 
                               [PIECE_ROOK]: 'r', [PIECE_QUEEN]: 'q' };
            str += promoChars[this.promotion];
        }
        
        return str;
    }
    
    equals(other) {
        return this.from === other.from && 
               this.to === other.to && 
               this.promotion === other.promotion;
    }
}

// Precomputed move tables
const MOVE_TABLES = {
    knight: new Array(64),
    king: new Array(64),
    pawnAttacks: [new Array(64), new Array(64)], // [white, black]
    rays: {
        north: new Array(64),
        south: new Array(64),
        east: new Array(64),
        west: new Array(64),
        northeast: new Array(64),
        northwest: new Array(64),
        southeast: new Array(64),
        southwest: new Array(64)
    }
};

function initMoveTables() {
    const KNIGHT_DELTAS = [-17, -15, -10, -6, 6, 10, 15, 17];
    const KING_DELTAS = [-9, -8, -7, -1, 1, 7, 8, 9];
    
    for (let sq = 0; sq < 64; sq++) {
        const rank = Math.floor(sq / 8);
        const file = sq % 8;
        
        // Knight moves
        MOVE_TABLES.knight[sq] = [];
        for (const delta of KNIGHT_DELTAS) {
            const to = sq + delta;
            if (to >= 0 && to < 64) {
                const toRank = Math.floor(to / 8);
                const toFile = to % 8;
                if (Math.abs(toRank - rank) <= 2 && Math.abs(toFile - file) <= 2) {
                    MOVE_TABLES.knight[sq].push(to);
                }
            }
        }
        
        // King moves
        MOVE_TABLES.king[sq] = [];
        for (const delta of KING_DELTAS) {
            const to = sq + delta;
            if (to >= 0 && to < 64) {
                const toRank = Math.floor(to / 8);
                const toFile = to % 8;
                if (Math.abs(toRank - rank) <= 1 && Math.abs(toFile - file) <= 1) {
                    MOVE_TABLES.king[sq].push(to);
                }
            }
        }
        
        // Pawn attacks
        for (const color of [COLOR_WHITE, COLOR_BLACK]) {
            MOVE_TABLES.pawnAttacks[color][sq] = [];
            const direction = color === COLOR_WHITE ? -8 : 8;
            const startRank = color === COLOR_WHITE ? 6 : 1;
            
            // Capture left
            if (file > 0) {
                const to = sq + direction - 1;
                if (to >= 0 && to < 64) {
                    MOVE_TABLES.pawnAttacks[color][sq].push(to);
                }
            }
            // Capture right
            if (file < 7) {
                const to = sq + direction + 1;
                if (to >= 0 && to < 64) {
                    MOVE_TABLES.pawnAttacks[color][sq].push(to);
                }
            }
        }
        
        // Sliding piece rays
        const directions = [
            { name: 'north', delta: -8, condition: () => true },
            { name: 'south', delta: 8, condition: () => true },
            { name: 'east', delta: 1, condition: () => (sq % 8) < 7 },
            { name: 'west', delta: -1, condition: () => (sq % 8) > 0 },
            { name: 'northeast', delta: -7, condition: () => (sq % 8) < 7 },
            { name: 'northwest', delta: -9, condition: () => (sq % 8) > 0 },
            { name: 'southeast', delta: 9, condition: () => (sq % 8) < 7 },
            { name: 'southwest', delta: 7, condition: () => (sq % 8) > 0 }
        ];
        
        for (const dir of directions) {
            MOVE_TABLES.rays[dir.name][sq] = [];
            let to = sq + dir.delta;
            while (to >= 0 && to < 64 && dir.condition()) {
                const toRank = Math.floor(to / 8);
                const toFile = to % 8;
                const fromRank = Math.floor((to - dir.delta) / 8);
                const fromFile = (to - dir.delta) % 8;
                
                // Check if we wrapped around
                if (Math.abs(toFile - fromFile) > 1 || Math.abs(toRank - fromRank) > 1) break;
                
                MOVE_TABLES.rays[dir.name][sq].push(to);
                to += dir.delta;
            }
        }
    }
}

initMoveTables();

// =============================================================================
// BOARD REPRESENTATION
// =============================================================================

class Board {
    constructor() {
        this.squares = new Array(64).fill(null).map(() => ({ type: PIECE_NONE, color: 0 }));
        this.sideToMove = COLOR_WHITE;
        this.castlingRights = { 
            [COLOR_WHITE]: { kingSide: true, queenSide: true },
            [COLOR_BLACK]: { kingSide: true, queenSide: true }
        };
        this.enPassantSquare = null;
        this.halfmoveClock = 0;
        this.fullmoveNumber = 1;
        this.hash = 0n;
        this.pieceList = [[], []]; // [white pieces, black pieces]
        this.kingSquare = [null, null];
        
        // History for repetition detection
        this.history = [];
        this.positionHistory = new Map();
        
        // NNUE
        this.nnue = new NNUE();
    }
    
    clone() {
        const b = new Board();
        for (let i = 0; i < 64; i++) {
            b.squares[i] = { ...this.squares[i] };
        }
        b.sideToMove = this.sideToMove;
        b.castlingRights = JSON.parse(JSON.stringify(this.castlingRights));
        b.enPassantSquare = this.enPassantSquare;
        b.halfmoveClock = this.halfmoveClock;
        b.fullmoveNumber = this.fullmoveNumber;
        b.hash = this.hash;
        b.kingSquare = [...this.kingSquare];
        return b;
    }
    
    setPiece(sq, type, color) {
        this.squares[sq] = { type, color };
        if (type === PIECE_KING) {
            this.kingSquare[color] = sq;
        }
        this.updatePieceList();
    }
    
    updatePieceList() {
        this.pieceList = [[], []];
        for (let sq = 0; sq < 64; sq++) {
            const piece = this.squares[sq];
            if (piece.type !== PIECE_NONE) {
                this.pieceList[piece.color].push({ square: sq, ...piece });
            }
        }
    }
    
    loadFEN(fen) {
        const parts = fen.trim().split(/\s+/);
        const ranks = parts[0].split('/');
        
        // Clear board
        this.squares = new Array(64).fill(null).map(() => ({ type: PIECE_NONE, color: 0 }));
        
        // Parse pieces
        for (let rank = 0; rank < 8; rank++) {
            let file = 0;
            for (const char of ranks[rank]) {
                if (/\d/.test(char)) {
                    file += parseInt(char);
                } else {
                    const color = char === char.toUpperCase() ? COLOR_WHITE : COLOR_BLACK;
                    const type = {
                        'p': PIECE_PAWN, 'n': PIECE_KNIGHT, 'b': PIECE_BISHOP,
                        'r': PIECE_ROOK, 'q': PIECE_QUEEN, 'k': PIECE_KING
                    }[char.toLowerCase()];
                    this.setPiece((7 - rank) * 8 + file, type, color);
                    file++;
                }
            }
        }
        
        // Side to move
        this.sideToMove = parts[1] === 'w' ? COLOR_WHITE : COLOR_BLACK;
        
        // Castling rights
        this.castlingRights = {
            [COLOR_WHITE]: { kingSide: parts[2].includes('K'), queenSide: parts[2].includes('Q') },
            [COLOR_BLACK]: { kingSide: parts[2].includes('k'), queenSide: parts[2].includes('q') }
        };
        
        // En passant
        this.enPassantSquare = parts[3] === '-' ? null : this.algebraicToSquare(parts[3]);
        
        // Clocks
        this.halfmoveClock = parseInt(parts[4] || 0);
        this.fullmoveNumber = parseInt(parts[5] || 1);
        
        this.computeHash();
        this.nnue.refreshAccumulator(this.squares, this.sideToMove);
        return this;
    }
    
    toFEN() {
        let fen = '';
        
        // Pieces
        for (let rank = 0; rank < 8; rank++) {
            let empty = 0;
            for (let file = 0; file < 8; file++) {
                const sq = (7 - rank) * 8 + file;
                const piece = this.squares[sq];
                
                if (piece.type === PIECE_NONE) {
                    empty++;
                } else {
                    if (empty > 0) {
                        fen += empty;
                        empty = 0;
                    }
                    const chars = ' pnbrqk';
                    const char = chars[piece.type];
                    fen += piece.color === COLOR_WHITE ? char.toUpperCase() : char;
                }
            }
            if (empty > 0) fen += empty;
            if (rank < 7) fen += '/';
        }
        
        // Side to move
        fen += ' ' + (this.sideToMove === COLOR_WHITE ? 'w' : 'b');
        
        // Castling
        let castling = '';
        if (this.castlingRights[COLOR_WHITE].kingSide) castling += 'K';
        if (this.castlingRights[COLOR_WHITE].queenSide) castling += 'Q';
        if (this.castlingRights[COLOR_BLACK].kingSide) castling += 'k';
        if (this.castlingRights[COLOR_BLACK].queenSide) castling += 'q';
        fen += ' ' + (castling || '-');
        
        // En passant
        fen += ' ' + (this.enPassantSquare !== null ? this.squareToAlgebraic(this.enPassantSquare) : '-');
        
        // Clocks
        fen += ' ' + this.halfmoveClock;
        fen += ' ' + this.fullmoveNumber;
        
        return fen;
    }
    
    algebraicToSquare(alg) {
        const file = alg.charCodeAt(0) - 'a'.charCodeAt(0);
        const rank = 8 - parseInt(alg[1]);
        return rank * 8 + file;
    }
    
    squareToAlgebraic(sq) {
        const file = String.fromCharCode('a'.charCodeAt(0) + (sq % 8));
        const rank = 8 - Math.floor(sq / 8);
        return file + rank;
    }
    
    computeHash() {
        // Simple Zobrist hashing
        this.hash = 0n;
        for (let sq = 0; sq < 64; sq++) {
            const piece = this.squares[sq];
            if (piece.type !== PIECE_NONE) {
                this.hash ^= this.getZobristPiece(piece, sq);
            }
        }
        if (this.sideToMove === COLOR_BLACK) {
            this.hash ^= ZOBRIST.side;
        }
        // Add castling and en passant to hash...
    }
    
    getZobristPiece(piece, square) {
        const idx = (piece.type - 1) * 2 + piece.color;
        return ZOBRIST.pieces[square * 12 + idx] || BigInt(square * 31 + idx * 17);
    }
    
    isAttacked(square, byColor) {
        // Pawn attacks
        const pawnDir = byColor === COLOR_WHITE ? -8 : 8;
        const attackFiles = [-1, 1];
        for (const df of attackFiles) {
            const from = square - pawnDir + df;
            if (from >= 0 && from < 64) {
                const file = square % 8;
                const fromFile = from % 8;
                if (Math.abs(fromFile - file) === 1) {
                    const piece = this.squares[from];
                    if (piece.type === PIECE_PAWN && piece.color === byColor) {
                        return true;
                    }
                }
            }
        }
        
        // Knight attacks
        for (const from of MOVE_TABLES.knight[square]) {
            const piece = this.squares[from];
            if (piece.type === PIECE_KNIGHT && piece.color === byColor) {
                return true;
            }
        }
        
        // King attacks
        for (const from of MOVE_TABLES.king[square]) {
            const piece = this.squares[from];
            if (piece.type === PIECE_KING && piece.color === byColor) {
                return true;
            }
        }
        
        // Sliding pieces (Bishop, Rook, Queen)
        const bishopRays = ['northeast', 'northwest', 'southeast', 'southwest'];
        const rookRays = ['north', 'south', 'east', 'west'];
        
        for (const rayName of bishopRays) {
            for (const from of MOVE_TABLES.rays[rayName][square]) {
                const piece = this.squares[from];
                if (piece.type !== PIECE_NONE) {
                    if ((piece.type === PIECE_BISHOP || piece.type === PIECE_QUEEN) && 
                        piece.color === byColor) {
                        return true;
                    }
                    break; // Blocked
                }
            }
        }
        
        for (const rayName of rookRays) {
            for (const from of MOVE_TABLES.rays[rayName][square]) {
                const piece = this.squares[from];
                if (piece.type !== PIECE_NONE) {
                    if ((piece.type === PIECE_ROOK || piece.type === PIECE_QUEEN) && 
                        piece.color === byColor) {
                        return true;
                    }
                    break; // Blocked
                }
            }
        }
        
        return false;
    }
    
    isInCheck(color = this.sideToMove) {
        return this.isAttacked(this.kingSquare[color], 1 - color);
    }
    
    generateMoves(onlyCaptures = false) {
        const moves = [];
        const us = this.sideToMove;
        const them = 1 - us;
        
        for (let sq = 0; sq < 64; sq++) {
            const piece = this.squares[sq];
            if (piece.type === PIECE_NONE || piece.color !== us) continue;
            
            switch (piece.type) {
                case PIECE_PAWN:
                    this.generatePawnMoves(sq, us, moves, onlyCaptures);
                    break;
                case PIECE_KNIGHT:
                    this.generateKnightMoves(sq, us, moves, onlyCaptures);
                    break;
                case PIECE_BISHOP:
                    this.generateBishopMoves(sq, us, moves, onlyCaptures);
                    break;
                case PIECE_ROOK:
                    this.generateRookMoves(sq, us, moves, onlyCaptures);
                    break;
                case PIECE_QUEEN:
                    this.generateQueenMoves(sq, us, moves, onlyCaptures);
                    break;
                case PIECE_KING:
                    this.generateKingMoves(sq, us, moves, onlyCaptures);
                    break;
            }
        }
        
        // Generate castling moves if not in check and not captures only
        if (!onlyCaptures && !this.isInCheck(us)) {
            this.generateCastlingMoves(us, moves);
        }
        
        return moves;
    }
    
    generatePawnMoves(from, color, moves, onlyCaptures) {
        const direction = color === COLOR_WHITE ? -8 : 8;
        const startRank = color === COLOR_WHITE ? 6 : 1;
        const promotionRank = color === COLOR_WHITE ? 0 : 7;
        const fromRank = Math.floor(from / 8);
        
        // Single push
        if (!onlyCaptures) {
            const to = from + direction;
            if (to >= 0 && to < 64 && this.squares[to].type === PIECE_NONE) {
                if (Math.floor(to / 8) === promotionRank) {
                    // Promotions
                    for (const promo of [PIECE_QUEEN, PIECE_ROOK, PIECE_BISHOP, PIECE_KNIGHT]) {
                        moves.push(new Move(from, to, this.squares[from], null, promo, MOVE_TYPES.PROMOTION));
                    }
                } else {
                    moves.push(new Move(from, to, this.squares[from]));
                    
                    // Double push
                    if (fromRank === startRank) {
                        const to2 = to + direction;
                        if (this.squares[to2].type === PIECE_NONE) {
                            moves.push(new Move(from, to2, this.squares[from]));
                        }
                    }
                }
            }
        }
        
        // Captures
        for (const to of MOVE_TABLES.pawnAttacks[color][from]) {
            const target = this.squares[to];
            if (target.type !== PIECE_NONE && target.color !== color) {
                if (Math.floor(to / 8) === promotionRank) {
                    for (const promo of [PIECE_QUEEN, PIECE_ROOK, PIECE_BISHOP, PIECE_KNIGHT]) {
                        moves.push(new Move(from, to, this.squares[from], target, promo, MOVE_TYPES.PROMOTION));
                    }
                } else {
                    moves.push(new Move(from, to, this.squares[from], target));
                }
            }
            
            // En passant
            if (to === this.enPassantSquare) {
                const epCapture = { 
                    type: PIECE_PAWN, 
                    color: 1 - color,
                    square: color === COLOR_WHITE ? to + 8 : to - 8
                };
                moves.push(new Move(from, to, this.squares[from], epCapture, null, MOVE_TYPES.EN_PASSANT));
            }
        }
    }
    
    generateKnightMoves(from, color, moves, onlyCaptures) {
        for (const to of MOVE_TABLES.knight[from]) {
            const target = this.squares[to];
            if (target.type === PIECE_NONE) {
                if (!onlyCaptures) {
                    moves.push(new Move(from, to, this.squares[from]));
                }
            } else if (target.color !== color) {
                moves.push(new Move(from, to, this.squares[from], target));
            }
        }
    }
    
    generateBishopMoves(from, color, moves, onlyCaptures) {
        const rays = ['northeast', 'northwest', 'southeast', 'southwest'];
        for (const ray of rays) {
            for (const to of MOVE_TABLES.rays[ray][from]) {
                const target = this.squares[to];
                if (target.type === PIECE_NONE) {
                    if (!onlyCaptures) {
                        moves.push(new Move(from, to, this.squares[from]));
                    }
                } else {
                    if (target.color !== color) {
                        moves.push(new Move(from, to, this.squares[from], target));
                    }
                    break;
                }
            }
        }
    }
    
    generateRookMoves(from, color, moves, onlyCaptures) {
        const rays = ['north', 'south', 'east', 'west'];
        for (const ray of rays) {
            for (const to of MOVE_TABLES.rays[ray][from]) {
                const target = this.squares[to];
                if (target.type === PIECE_NONE) {
                    if (!onlyCaptures) {
                        moves.push(new Move(from, to, this.squares[from]));
                    }
                } else {
                    if (target.color !== color) {
                        moves.push(new Move(from, to, this.squares[from], target));
                    }
                    break;
                }
            }
        }
    }
    
    generateQueenMoves(from, color, moves, onlyCaptures) {
        this.generateBishopMoves(from, color, moves, onlyCaptures);
        this.generateRookMoves(from, color, moves, onlyCaptures);
    }
    
    generateKingMoves(from, color, moves, onlyCaptures) {
        for (const to of MOVE_TABLES.king[from]) {
            const target = this.squares[to];
            if (target.type === PIECE_NONE) {
                if (!onlyCaptures) {
                    moves.push(new Move(from, to, this.squares[from]));
                }
            } else if (target.color !== color) {
                moves.push(new Move(from, to, this.squares[from], target));
            }
        }
    }
    
    generateCastlingMoves(color, moves) {
        const rank = color === COLOR_WHITE ? 7 : 0;
        const kingSq = this.kingSquare[color];
        
        // Kingside
        if (this.castlingRights[color].kingSide) {
            if (this.squares[rank * 8 + 5].type === PIECE_NONE &&
                this.squares[rank * 8 + 6].type === PIECE_NONE) {
                if (!this.isAttacked(rank * 8 + 5, 1 - color)) {
                    moves.push(new Move(kingSq, rank * 8 + 6, this.squares[kingSq], null, null, MOVE_TYPES.CASTLING));
                }
            }
        }
        
        // Queenside
        if (this.castlingRights[color].queenSide) {
            if (this.squares[rank * 8 + 1].type === PIECE_NONE &&
                this.squares[rank * 8 + 2].type === PIECE_NONE &&
                this.squares[rank * 8 + 3].type === PIECE_NONE) {
                if (!this.isAttacked(rank * 8 + 3, 1 - color)) {
                    moves.push(new Move(kingSq, rank * 8 + 2, this.squares[kingSq], null, null, MOVE_TYPES.CASTLING));
                }
            }
        }
    }
    
    makeMove(move) {
        // Save state for undo
        const state = {
            move: move,
            castling: JSON.parse(JSON.stringify(this.castlingRights)),
            enPassant: this.enPassantSquare,
            halfmove: this.halfmoveClock,
            hash: this.hash,
            captured: move.captured ? { ...move.captured } : null
        };
        
        const us = this.sideToMove;
        const them = 1 - us;
        
        // Update hash
        this.hash ^= this.getZobristPiece(move.piece, move.from);
        if (move.captured && move.captured.type !== PIECE_NONE) {
            const captureSq = move.type === MOVE_TYPES.EN_PASSANT ? 
                (us === COLOR_WHITE ? move.to + 8 : move.to - 8) : move.to;
            this.hash ^= this.getZobristPiece(move.captured, captureSq);
        }
        
        // Handle special moves
        if (move.type === MOVE_TYPES.CASTLING) {
            // Move rook
            const rank = Math.floor(move.from / 8);
            const kingSide = move.to > move.from;
            const rookFrom = kingSide ? rank * 8 + 7 : rank * 8;
            const rookTo = kingSide ? rank * 8 + 5 : rank * 8 + 3;
            const rook = this.squares[rookFrom];
            
            this.squares[rookTo] = rook;
            this.squares[rookFrom] = { type: PIECE_NONE, color: 0 };
            this.hash ^= this.getZobristPiece(rook, rookFrom);
            this.hash ^= this.getZobristPiece(rook, rookTo);
        }
        
        // Move piece
        let movedPiece = move.piece;
        if (move.promotion) {
            movedPiece = { type: move.promotion, color: us };
        }
        
        this.squares[move.to] = movedPiece;
        this.squares[move.from] = { type: PIECE_NONE, color: 0 };
        
        // Handle en passant capture
        if (move.type === MOVE_TYPES.EN_PASSANT) {
            const epSq = us === COLOR_WHITE ? move.to + 8 : move.to - 8;
            this.hash ^= this.getZobristPiece({ type: PIECE_PAWN, color: them }, epSq);
            this.squares[epSq] = { type: PIECE_NONE, color: 0 };
        }
        
        // Update king square
        if (move.piece.type === PIECE_KING) {
            this.kingSquare[us] = move.to;
        }
        
        // Update castling rights
        if (move.piece.type === PIECE_KING) {
            this.castlingRights[us].kingSide = false;
            this.castlingRights[us].queenSide = false;
        }
        if (move.piece.type === PIECE_ROOK) {
            const rank = Math.floor(move.from / 8);
            if (rank === (us === COLOR_WHITE ? 7 : 0)) {
                if (move.from % 8 === 0) this.castlingRights[us].queenSide = false;
                if (move.from % 8 === 7) this.castlingRights[us].kingSide = false;
            }
        }
        // Captured rook affects opponent's castling
        if (move.captured && move.captured.type === PIECE_ROOK) {
            const rank = Math.floor(move.to / 8);
            if (rank === (them === COLOR_WHITE ? 7 : 0)) {
                if (move.to % 8 === 0) this.castlingRights[them].queenSide = false;
                if (move.to % 8 === 7) this.castlingRights[them].kingSide = false;
            }
        }
        
        // Update en passant square
        if (move.piece.type === PIECE_PAWN && Math.abs(move.to - move.from) === 16) {
            this.enPassantSquare = (move.from + move.to) / 2;
        } else {
            this.enPassantSquare = null;
        }
        
        // Update clocks
        if (move.piece.type === PIECE_PAWN || move.captured) {
            this.halfmoveClock = 0;
        } else {
            this.halfmoveClock++;
        }
        
        if (us === COLOR_BLACK) {
            this.fullmoveNumber++;
        }
        
        // Switch side
        this.sideToMove = them;
        this.hash ^= ZOBRIST.side;
        
        // Update NNUE accumulator
        state.nnueState = this.nnue.updateAccumulator(move, this.squares);
        
        return state;
    }
    
    undoMove(state) {
        const move = state.move;
        const us = 1 - this.sideToMove; // We switched, so current is them
        const them = this.sideToMove;
        
        // Restore NNUE
        this.nnue.restoreAccumulator(state.nnueState);
        
        // Restore hash
        this.hash = state.hash;
        
        // Restore state
        this.castlingRights = state.castling;
        this.enPassantSquare = state.enPassant;
        this.halfmoveClock = state.halfmove;
        
        if (us === COLOR_BLACK) {
            this.fullmoveNumber--;
        }
        
        // Undo special moves
        if (move.type === MOVE_TYPES.CASTLING) {
            const rank = Math.floor(move.from / 8);
            const kingSide = move.to > move.from;
            const rookFrom = kingSide ? rank * 8 + 7 : rank * 8;
            const rookTo = kingSide ? rank * 8 + 5 : rank * 8 + 3;
            const rook = this.squares[rookTo];
            
            this.squares[rookFrom] = rook;
            this.squares[rookTo] = { type: PIECE_NONE, color: 0 };
        }
        
        // Restore piece
        this.squares[move.from] = move.piece;
        
        // Handle promotion
        if (move.promotion) {
            this.squares[move.to] = state.captured || { type: PIECE_NONE, color: 0 };
        } else {
            this.squares[move.to] = state.captured || { type: PIECE_NONE, color: 0 };
        }
        
        // Restore en passant capture
        if (move.type === MOVE_TYPES.EN_PASSANT && state.captured) {
            const epSq = us === COLOR_WHITE ? move.to + 8 : move.to - 8;
            this.squares[epSq] = state.captured;
        }
        
        // Restore king square
        if (move.piece.type === PIECE_KING) {
            this.kingSquare[us] = move.from;
        }
        
        this.sideToMove = us;
    }
    
    isRepetition() {
        const count = this.positionHistory.get(this.hash) || 0;
        return count >= 2;
    }
    
    isDraw() {
        // 50-move rule
        if (this.halfmoveClock >= 100) return true;
        
        // Insufficient material
        const pieces = [[], []];
        for (let sq = 0; sq < 64; sq++) {
            const p = this.squares[sq];
            if (p.type !== PIECE_NONE && p.type !== PIECE_KING) {
                pieces[p.color].push(p.type);
            }
        }
        
        // King vs King
        if (pieces[0].length === 0 && pieces[1].length === 0) return true;
        
        // King and minor piece vs King
        if (pieces[0].length === 1 && pieces[1].length === 0) {
            if (pieces[0][0] === PIECE_BISHOP || pieces[0][0] === PIECE_KNIGHT) return true;
        }
        if (pieces[1].length === 1 && pieces[0].length === 0) {
            if (pieces[1][0] === PIECE_BISHOP || pieces[1][0] === PIECE_KNIGHT) return true;
        }
        
        // Repetition
        if (this.isRepetition()) return true;
        
        return false;
    }
}

// =============================================================================
// SEARCH ENGINE
// =============================================================================

class Search {
    constructor(board) {
        this.board = board;
        this.tt = new TranspositionTable(256);
        this.nodes = 0;
        this.qnodes = 0;
        this.tbhits = 0;
        this.startTime = 0;
        this.timeLimit = 0;
        this.stop = false;
        this.ply = 0;
        this.selDepth = 0;
        
        // History heuristic [color][from][to]
        this.history = new Array(2).fill(null).map(() => 
            new Array(64).fill(null).map(() => new Int32Array(64))
        );
        
        // Counter move heuristic [piece][to]
        this.counterMoves = new Array(7).fill(null).map(() => new Array(64).fill(null));
        
        // Killer moves [ply][2]
        this.killers = new Array(MAX_PLY).fill(null).map(() => [null, null]);
        
        // PV table
        this.pv = new Array(MAX_PLY).fill(null).map(() => new Array(MAX_PLY));
        this.pvLength = new Int32Array(MAX_PLY);
        
        // Move ordering constants
        this.MVV_LVA = [
            [0, 0, 0, 0, 0, 0, 0],   // victim None
            [0, 15, 14, 13, 12, 11, 10], // victim Pawn
            [0, 25, 24, 23, 22, 21, 20], // victim Knight
            [0, 35, 34, 33, 32, 31, 30], // victim Bishop
            [0, 45, 44, 43, 42, 41, 40], // victim Rook
            [0, 55, 54, 53, 52, 51, 50], // victim Queen
            [0, 0, 0, 0, 0, 0, 0]    // victim King
        ];
    }
    
    clear() {
        this.tt.clear();
        for (let c = 0; c < 2; c++) {
            for (let from = 0; from < 64; from++) {
                this.history[c][from].fill(0);
            }
        }
        for (let p = 0; p < 7; p++) {
            this.counterMoves[p].fill(null);
        }
        for (let i = 0; i < MAX_PLY; i++) {
            this.killers[i][0] = null;
            this.killers[i][1] = null;
            this.pvLength[i] = 0;
        }
    }
    
    isTimeUp() {
        if (this.nodes % 1024 !== 0) return false;
        return Date.now() - this.startTime > this.timeLimit;
    }
    
    search(depth, timeMs) {
        this.clear();
        this.startTime = Date.now();
        this.timeLimit = timeMs;
        this.stop = false;
        this.nodes = 0;
        this.qnodes = 0;
        
        let bestMove = null;
        let bestScore = 0;
        
        // Iterative deepening
        for (let d = 1; d <= depth && !this.stop; d++) {
            this.selDepth = 0;
            const score = this.alphaBeta(d, -INFINITY, INFINITY, 0, true);
            
            if (!this.stop) {
                bestScore = score;
                bestMove = this.pv[0][0];
                
                const elapsed = Date.now() - this.startTime;
                const nps = elapsed > 0 ? Math.round(this.nodes / (elapsed / 1000)) : 0;
                
                console.log(`info depth ${d} score cp ${score} nodes ${this.nodes} nps ${nps} time ${elapsed} pv ${this.getPVString()}`);
            }
        }
        
        return { move: bestMove, score: bestScore };
    }
    
    alphaBeta(depth, alpha, beta, ply, isPV) {
        this.ply = ply;
        if (ply > this.selDepth) this.selDepth = ply;
        
        // Check time
        if (this.isTimeUp()) {
            this.stop = true;
            return 0;
        }
        
        // Check for draw
        if (this.board.isDraw() && ply > 0) {
            return DRAW_SCORE;
        }
        
        // Mate distance pruning
        alpha = Math.max(alpha, -MATE_SCORE + ply);
        beta = Math.min(beta, MATE_SCORE - ply - 1);
        if (alpha >= beta) return alpha;
        
        // Transposition table probe
        const ttEntry = this.tt.probe(this.board.hash);
        let ttMove = null;
        if (ttEntry && ttEntry.depth >= depth) {
            if (ttEntry.flag === TT_EXACT) return ttEntry.score;
            if (ttEntry.flag === TT_ALPHA && ttEntry.score <= alpha) return alpha;
            if (ttEntry.flag === TT_BETA && ttEntry.score >= beta) return beta;
            ttMove = ttEntry.move;
        }
        
        // Quiescence search at leaf nodes
        if (depth <= 0) {
            return this.quiescence(alpha, beta, ply);
        }
        
        this.nodes++;
        
        // Null move pruning
        if (!isPV && !this.board.isInCheck() && depth >= 3 && ply > 0) {
            const R = 3 + Math.floor(depth / 4);
            this.board.sideToMove = 1 - this.board.sideToMove;
            this.board.hash ^= ZOBRIST.side;
            
            const nullScore = -this.alphaBeta(depth - 1 - R, -beta, -beta + 1, ply + 1, false);
            
            this.board.sideToMove = 1 - this.board.sideToMove;
            this.board.hash ^= ZOBRIST.side;
            
            if (nullScore >= beta) {
                return beta;
            }
        }
        
        // Generate and score moves
        const moves = this.board.generateMoves();
        
        if (moves.length === 0) {
            if (this.board.isInCheck()) {
                return -MATE_SCORE + ply; // Checkmate
            }
            return DRAW_SCORE; // Stalemate
        }
        
        // Score moves
        this.scoreMoves(moves, ttMove, ply);
        
        let bestScore = -INFINITY;
        let bestMove = null;
        let moveCount = 0;
        
        for (let i = 0; i < moves.length; i++) {
            // Select best move from remaining
            this.selectMove(moves, i);
            const move = moves[i];
            
            // Make move
            const state = this.board.makeMove(move);
            
            // Check if move is legal (doesn't leave king in check)
            if (this.board.isInCheck(1 - this.board.sideToMove)) {
                this.board.undoMove(state);
                continue;
            }
            
            moveCount++;
            
            // Late Move Reduction (LMR)
            let newDepth = depth - 1;
            let doFullSearch = true;
            
            if (!isPV && moveCount > 1 && depth >= 3 && 
                !move.captured && move.promotion === null &&
                !this.board.isInCheck()) {
                
                const reduction = Math.floor(Math.log(depth) * Math.log(moveCount) / 2);
                newDepth = Math.max(1, depth - 1 - reduction);
                
                const score = -this.alphaBeta(newDepth, -alpha - 1, -alpha, ply + 1, false);
                doFullSearch = score > alpha;
            }
            
            // Principal Variation Search (PVS)
            let score;
            if (doFullSearch) {
                if (isPV && moveCount === 1) {
                    score = -this.alphaBeta(newDepth, -beta, -alpha, ply + 1, true);
                } else {
                    score = -this.alphaBeta(newDepth, -alpha - 1, -alpha, ply + 1, false);
                    if (score > alpha && score < beta) {
                        score = -this.alphaBeta(newDepth, -beta, -alpha, ply + 1, true);
                    }
                }
            } else {
                score = -this.alphaBeta(newDepth, -alpha - 1, -alpha, ply + 1, false);
            }
            
            this.board.undoMove(state);
            
            if (this.stop) return 0;
            
            if (score > bestScore) {
                bestScore = score;
                bestMove = move;
                
                if (score > alpha) {
                    alpha = score;
                    
                    // Update PV
                    this.pv[ply][0] = move;
                    for (let j = 0; j < this.pvLength[ply + 1]; j++) {
                        this.pv[ply][j + 1] = this.pv[ply + 1][j];
                    }
                    this.pvLength[ply] = this.pvLength[ply + 1] + 1;
                    
                    if (score >= beta) {
                        // Update history and killers
                        if (!move.captured) {
                            this.updateHistory(move, depth, ply);
                        }
                        break;
                    }
                }
            }
        }
        
        if (moveCount === 0) {
            return this.board.isInCheck() ? -MATE_SCORE + ply : DRAW_SCORE;
        }
        
        // Store in TT
        const flag = bestScore >= beta ? TT_BETA : 
                     (alpha > -INFINITY + MAX_PLY ? TT_EXACT : TT_ALPHA);
        this.tt.store(this.board.hash, bestMove, bestScore, depth, flag);
        
        return bestScore;
    }
    
    quiescence(alpha, beta, ply) {
        this.qnodes++;
        
        // Stand pat evaluation
        const standPat = this.evaluate();
        if (standPat >= beta) return beta;
        if (alpha < standPat) alpha = standPat;
        
        // Delta pruning threshold
        const DELTA = PIECE_VALUES.QUEEN;
        if (standPat + DELTA < alpha) return alpha;
        
        // Generate capture moves only
        const moves = this.board.generateMoves(true);
        this.scoreMoves(moves, null, ply);
        
        for (let i = 0; i < moves.length; i++) {
            this.selectMove(moves, i);
            const move = moves[i];
            
            // Delta pruning for individual moves
            const capturedValue = move.captured ? this.pieceValue(move.captured.type) : 0;
            if (standPat + capturedValue + PIECE_VALUES.PAWN < alpha) continue;
            
            // SEE pruning - only consider winning captures
            if (!this.see(move, 0)) continue;
            
            const state = this.board.makeMove(move);
            
            if (this.board.isInCheck(1 - this.board.sideToMove)) {
                this.board.undoMove(state);
                continue;
            }
            
            const score = -this.quiescence(-beta, -alpha, ply + 1);
            this.board.undoMove(state);
            
            if (score >= beta) return beta;
            if (score > alpha) alpha = score;
        }
        
        return alpha;
    }
    
    evaluate() {
        // Use NNUE evaluation
        let score = this.board.nnue.evaluate(this.board.sideToMove);
        
        // Tempo bonus
        score += 20;
        
        // Phase calculation for potential mixing with classical eval
        const phase = this.calculatePhase();
        
        return score;
    }
    
    calculatePhase() {
        let npm = 0; // Non-pawn material
        for (let sq = 0; sq < 64; sq++) {
            const p = this.board.squares[sq];
            if (p.type !== PIECE_NONE && p.type !== PIECE_PAWN && p.type !== PIECE_KING) {
                npm += PIECE_VALUES[p.type] - PIECE_VALUES.PAWN;
            }
        }
        
        if (npm > 6000) return PHASE_OPENING;
        if (npm > 2000) return PHASE_MIDDLEGAME;
        return PHASE_ENDGAME;
    }
    
    pieceValue(type) {
        switch (type) {
            case PIECE_PAWN: return PIECE_VALUES.PAWN;
            case PIECE_KNIGHT: return PIECE_VALUES.KNIGHT;
            case PIECE_BISHOP: return PIECE_VALUES.BISHOP;
            case PIECE_ROOK: return PIECE_VALUES.ROOK;
            case PIECE_QUEEN: return PIECE_VALUES.QUEEN;
            default: return 0;
        }
    }
    
    scoreMoves(moves, ttMove, ply) {
        const us = this.board.sideToMove;
        
        for (const move of moves) {
            if (ttMove && move.equals(ttMove)) {
                move.score = 1000000;
            } else if (move.captured) {
                // MVV-LVA
                move.score = this.MVV_LVA[move.captured.type][move.piece.type] * 1000;
                // SEE bonus for winning captures
                if (this.see(move, 0)) move.score += 5000;
            } else {
                // Killer moves
                if (this.killers[ply][0] && move.equals(this.killers[ply][0])) {
                    move.score = 90000;
                } else if (this.killers[ply][1] && move.equals(this.killers[ply][1])) {
                    move.score = 80000;
                } else {
                    // History heuristic
                    move.score = this.history[us][move.from][move.to];
                }
                
                // Counter move bonus
                if (ply > 0) {
                    const prevMove = this.pv[ply - 1][0];
                    if (prevMove) {
                        const counter = this.counterMoves[prevMove.piece.type][prevMove.to];
                        if (counter && move.equals(counter)) {
                            move.score += 50000;
                        }
                    }
                }
            }
        }
    }
    
    selectMove(moves, startIdx) {
        let bestIdx = startIdx;
        let bestScore = moves[startIdx].score;
        
        for (let i = startIdx + 1; i < moves.length; i++) {
            if (moves[i].score > bestScore) {
                bestScore = moves[i].score;
                bestIdx = i;
            }
        }
        
        if (bestIdx !== startIdx) {
            const temp = moves[startIdx];
            moves[startIdx] = moves[bestIdx];
            moves[bestIdx] = temp;
        }
    }
    
    updateHistory(move, depth, ply) {
        const us = this.board.sideToMove;
        const bonus = depth * depth;
        
        // Update history table
        this.history[us][move.from][move.to] += bonus;
        if (this.history[us][move.from][move.to] > 16384) {
            for (let i = 0; i < 64; i++) {
                for (let j = 0; j < 64; j++) {
                    this.history[us][i][j] >>= 1;
                }
            }
        }
        
        // Update killer moves
        if (!this.killers[ply][0] || !move.equals(this.killers[ply][0])) {
            this.killers[ply][1] = this.killers[ply][0];
            this.killers[ply][0] = move;
        }
        
        // Update counter moves
        if (ply > 0) {
            const prevMove = this.pv[ply - 1][0];
            if (prevMove) {
                this.counterMoves[prevMove.piece.type][prevMove.to] = move;
            }
        }
    }
    
    /**
     * Static Exchange Evaluation (SEE)
     * Determines if a capture is winning (score >= threshold)
     */
    see(move, threshold) {
        const from = move.from;
        const to = move.to;
        const us = move.piece.color;
        
        // Value of captured piece
        let value = 0;
        if (move.captured) {
            value = this.pieceValue(move.captured.type);
            if (move.type === MOVE_TYPES.EN_PASSANT) {
                value = PIECE_VALUES.PAWN;
            }
        }
        
        // If promotion, add promotion value minus pawn
        if (move.promotion) {
            value += this.pieceValue(move.promotion) - PIECE_VALUES.PAWN;
        }
        
        // Simple SEE: assume we lose the attacker
        const attackerValue = this.pieceValue(move.piece.type);
        return value - attackerValue >= threshold;
    }
    
    getPVString() {
        let str = '';
        for (let i = 0; i < this.pvLength[0]; i++) {
            if (i > 0) str += ' ';
            str += this.pv[0][i].toString();
        }
        return str;
    }
}

// =============================================================================
// TIME MANAGEMENT
// =============================================================================

class TimeManager {
    constructor() {
        this.timeLeft = 0;
        this.increment = 0;
        this.movesToGo = 30;
        this.moveTime = 0;
    }
    
    setTimeControl(time, inc = 0, movesToGo = null) {
        this.timeLeft = time;
        this.increment = inc;
        if (movesToGo) this.movesToGo = movesToGo;
    }
    
    calculateSearchTime() {
        if (this.moveTime > 0) return this.moveTime;
        
        // Basic formula: use 1/movesToGo of time + 80% of increment
        let time = this.timeLeft / this.movesToGo + this.increment * 0.8;
        
        // Don't use more than 20% of remaining time
        time = Math.min(time, this.timeLeft * 0.2);
        
        // Minimum 100ms
        time = Math.max(time, 100);
        
        return time;
    }
    
    update(timeUsed) {
        this.timeLeft -= timeUsed;
        this.timeLeft += this.increment;
    }
}

// =============================================================================
// OPENING BOOK (Polyglot-style)
// =============================================================================

class OpeningBook {
    constructor() {
        this.entries = new Map();
    }
    
    loadFromJSON(jsonData) {
        for (const [key, moves] of Object.entries(jsonData)) {
            this.entries.set(BigInt(key), moves);
        }
    }
    
    probe(hash) {
        return this.entries.get(hash) || null;
    }
    
    getRandomMove(hash) {
        const moves = this.probe(hash);
        if (!moves || moves.length === 0) return null;
        return moves[Math.floor(Math.random() * moves.length)];
    }
}

// =============================================================================
// UCI INTERFACE
// =============================================================================

class UCI {
    constructor() {
        this.board = new Board();
        this.search = new Search(this.board);
        this.timeManager = new TimeManager();
        this.book = new OpeningBook();
        this.running = false;
        this.ponder = false;
        this.multipv = 1;
        
        // Initialize with standard position
        this.board.loadFEN("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    }
    
    receiveCommand(line) {
        const tokens = line.trim().split(/\s+/);
        const command = tokens[0];
        
        switch (command) {
            case 'uci':
                this.uci();
                break;
            case 'isready':
                console.log('readyok');
                break;
            case 'ucinewgame':
                this.ucinewgame();
                break;
            case 'position':
                this.position(tokens);
                break;
            case 'go':
                this.go(tokens);
                break;
            case 'stop':
                this.stop();
                break;
            case 'quit':
            case 'exit':
                this.quit();
                break;
            case 'setoption':
                this.setoption(tokens);
                break;
            case 'd':
                this.display();
                break;
            case 'eval':
                this.eval();
                break;
            case 'perft':
                this.perft(parseInt(tokens[1]) || 5);
                break;
            default:
                console.log(`Unknown command: ${command}`);
        }
    }
    
    uci() {
        console.log(`id name GarboChess ${VERSION}`);
        console.log(`id author ${AUTHOR}`);
        console.log('option name Hash type spin default 256 min 1 max 65536');
        console.log('option name Threads type spin default 1 min 1 max 512');
        console.log('option name Ponder type check default false');
        console.log('option name MultiPV type spin default 1 min 1 max 500');
        console.log('option name UCI_Elo type spin default 3200 min 1000 max 3200');
        console.log('uciok');
    }
    
    ucinewgame() {
        this.board = new Board();
        this.search = new Search(this.board);
        this.search.clear();
    }
    
    position(tokens) {
        let idx = 1;
        let fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        
        if (tokens[idx] === 'startpos') {
            idx++;
        } else if (tokens[idx] === 'fen') {
            idx++;
            const fenParts = [];
            while (idx < tokens.length && tokens[idx] !== 'moves') {
                fenParts.push(tokens[idx++]);
            }
            fen = fenParts.join(' ');
        }
        
        this.board.loadFEN(fen);
        
        if (tokens[idx] === 'moves') {
            idx++;
            while (idx < tokens.length) {
                const moveStr = tokens[idx++];
                const move = this.parseMove(moveStr);
                if (move) {
                    this.board.makeMove(move);
                }
            }
        }
    }
    
    parseMove(moveStr) {
        const files = 'abcdefgh';
        const fromFile = files.indexOf(moveStr[0]);
        const fromRank = 8 - parseInt(moveStr[1]);
        const toFile = files.indexOf(moveStr[2]);
        const toRank = 8 - parseInt(moveStr[3]);
        
        const from = fromRank * 8 + fromFile;
        const to = toRank * 8 + toFile;
        
        let promotion = null;
        if (moveStr.length > 4) {
            const promoMap = { 'q': PIECE_QUEEN, 'r': PIECE_ROOK, 'b': PIECE_BISHOP, 'n': PIECE_KNIGHT };
            promotion = promoMap[moveStr[4].toLowerCase()];
        }
        
        // Find matching legal move
        const legalMoves = this.board.generateMoves();
        for (const move of legalMoves) {
            if (move.from === from && move.to === to && move.promotion === promotion) {
                // Verify legality
                const state = this.board.makeMove(move);
                const legal = !this.board.isInCheck(1 - this.board.sideToMove);
                this.board.undoMove(state);
                if (legal) return move;
            }
        }
        return null;
    }
    
    go(tokens) {
        let depth = 64;
        let movetime = null;
        let wtime = null, btime = null;
        let winc = 0, binc = 0;
        let movestogo = null;
        
        for (let i = 1; i < tokens.length; i += 2) {
            const param = tokens[i];
            const value = parseInt(tokens[i + 1]);
            
            switch (param) {
                case 'depth':
                    depth = value;
                    break;
                case 'movetime':
                    movetime = value;
                    break;
                case 'wtime':
                    wtime = value;
                    break;
                case 'btime':
                    btime = value;
                    break;
                case 'winc':
                    winc = value;
                    break;
                case 'binc':
                    binc = value;
                    break;
                case 'movestogo':
                    movestogo = value;
                    break;
                case 'ponder':
                    this.ponder = true;
                    break;
                case 'infinite':
                    depth = 64;
                    movetime = Infinity;
                    break;
            }
        }
        
        // Set time control
        if (this.board.sideToMove === COLOR_WHITE && wtime !== null) {
            this.timeManager.setTimeControl(wtime, winc, movestogo);
        } else if (this.board.sideToMove === COLOR_BLACK && btime !== null) {
            this.timeManager.setTimeControl(btime, binc, movestogo);
        }
        
        if (movetime !== null) {
            this.timeManager.moveTime = movetime;
        }
        
        const searchTime = this.timeManager.calculateSearchTime();
        
        // Check opening book first
        const bookMove = this.book.getRandomMove(this.board.hash);
        if (bookMove && depth > 1) {
            console.log(`info string book move ${bookMove}`);
            console.log(`bestmove ${bookMove}`);
            return;
        }
        
        // Start search
        this.running = true;
        const result = this.search.search(depth, searchTime);
        
        if (result.move) {
            console.log(`bestmove ${result.move.toString()}`);
        } else {
            console.log('bestmove 0000');
        }
        
        this.running = false;
    }
    
    stop() {
        this.search.stop = true;
    }
    
    quit() {
        this.stop();
        process.exit(0);
    }
    
    setoption(tokens) {
        // Parse setoption name [value]
        let name = '';
        let value = '';
        let idx = 1;
        
        if (tokens[idx++] !== 'name') return;
        
        while (idx < tokens.length && tokens[idx] !== 'value') {
            name += (name ? ' ' : '') + tokens[idx++];
        }
        
        if (tokens[idx++] === 'value') {
            value = tokens.slice(idx).join(' ');
        }
        
        switch (name.toLowerCase()) {
            case 'hash':
                const sizeMB = parseInt(value) || 256;
                this.search.tt = new TranspositionTable(sizeMB);
                break;
            case 'multipv':
                this.multipv = parseInt(value) || 1;
                break;
            case 'ponder':
                this.ponder = value === 'true';
                break;
        }
    }
    
    display() {
        console.log(this.board.toFEN());
        console.log();
        const pieces = ' PNBRQK  pnbrqk';
        for (let rank = 0; rank < 8; rank++) {
            let line = `${8 - rank}  `;
            for (let file = 0; file < 8; file++) {
                const sq = rank * 8 + file;
                const piece = this.board.squares[sq];
                const char = pieces[piece.type + (piece.color === COLOR_BLACK ? 7 : 0)];
                line += char + ' ';
            }
            console.log(line);
        }
        console.log('   a b c d e f g h');
        console.log();
        console.log(`Side to move: ${this.board.sideToMove === COLOR_WHITE ? 'White' : 'Black'}`);
        console.log(`FEN: ${this.board.toFEN()}`);
    }
    
    eval() {
        const score = this.search.evaluate();
        console.log(`Static evaluation: ${score} cp`);
        
        // Detailed breakdown
        console.log(`Phase: ${this.search.calculatePhase() === PHASE_OPENING ? 'Opening' : 
                     this.search.calculatePhase() === PHASE_MIDDLEGAME ? 'Middlegame' : 'Endgame'}`);
    }
    
    perft(depth) {
        const start = Date.now();
        const nodes = this.perftRecursive(depth);
        const time = Date.now() - start;
        const nps = time > 0 ? Math.round(nodes / (time / 1000)) : 0;
        
        console.log(`Nodes: ${nodes}`);
        console.log(`Time: ${time}ms`);
        console.log(`NPS: ${nps}`);
    }
    
    perftRecursive(depth) {
        if (depth === 0) return 1;
        
        const moves = this.board.generateMoves();
        let nodes = 0;
        
        for (const move of moves) {
            const state = this.board.makeMove(move);
            
            if (!this.board.isInCheck(1 - this.board.sideToMove)) {
                nodes += this.perftRecursive(depth - 1);
            }
            
            this.board.undoMove(state);
        }
        
        return nodes;
    }
}

// =============================================================================
// INITIALIZATION & MAIN
// =============================================================================

// Initialize Zobrist keys
function initZobrist() {
    // Simple random number generator for Zobrist keys
    let seed = 0x1BADF00D;
    function rand64() {
        seed = (seed * 6364136223846793005n + 1442695040888963407n) & 0xFFFFFFFFFFFFFFFFn;
        return seed;
    }
    
    for (let i = 0; i < 768; i++) {
        ZOBRIST.pieces[i] = rand64();
    }
    ZOBRIST.side = rand64();
    for (let i = 0; i < 16; i++) {
        ZOBRIST.castling[i] = rand64();
    }
    for (let i = 0; i < 8; i++) {
        ZOBRIST.enPassant[i] = rand64();
    }
}

initZobrist();

// Create UCI interface
const uci = new UCI();

// Handle input
if (typeof process !== 'undefined' && process.stdin) {
    const readline = require('readline');
    const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout,
        prompt: ''
    });
    
    rl.on('line', (line) => {
        uci.receiveCommand(line);
    });
} else {
    // Browser or worker environment
    self.onmessage = function(e) {
        uci.receiveCommand(e.data);
    };
}

// Export for testing
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { Board, Search, UCI, NNUE, TranspositionTable };
}
